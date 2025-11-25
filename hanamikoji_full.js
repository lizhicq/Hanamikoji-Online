const http = require('http');
const url = require('url');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const PORT = 3000;

// --- Game Constants & Logic ---
const GEISHAS = [
    { id: 'pink', value: 2, color: '#FFB7C5', name: 'Ayame', image: '/data/images/ayame.jpg', favor: 0 },
    { id: 'green', value: 2, color: '#90EE90', name: 'Botan', image: '/data/images/botan.jpg', favor: 0 },
    { id: 'red', value: 2, color: '#FF6961', name: 'Chiyo', image: '/data/images/chiyo.jpg', favor: 0 },
    { id: 'yellow', value: 3, color: '#FDFD96', name: 'Daisy', image: '/data/images/daisy.jpg', favor: 0 },
    { id: 'purple', value: 3, color: '#E6E6FA', name: 'Eri', image: '/data/images/eri.jpg', favor: 0 },
    { id: 'peach', value: 4, color: '#FFDAB9', name: 'Fuji', image: '/data/images/fuji.jpg', favor: 0 },
    { id: 'blue', value: 5, color: '#ADD8E6', name: 'Ginko', image: '/data/images/ginko.jpg', favor: 0 }
];

const ACTIONS = {
    SECRET: 'secret',
    DISCARD: 'discard',
    GIFT: 'gift',
    COMPETITION: 'competition'
};

class HanamikojiGame {
    constructor(id, isPvE = false) {
        this.id = id;
        this.isPvE = isPvE;
        this.players = []; // { id, name, hand, actions, secret, discard, side }
        this.deck = [];
        this.removedCard = null;
        this.geishas = GEISHAS.map(g => ({ ...g, favor: null, items: { 0: [], 1: [] } }));
        this.round = 1;
        this.turnPlayerIdx = 0;
        this.phase = 'draw'; // draw, action, wait_response (for gift/comp), round_end, game_end
        this.pendingAction = null; // { type, sourcePlayerIdx, cards... }
        this.winner = null;
        this.logs = [];
        this.lastUpdateTime = Date.now();
    }

    addPlayer(id, name) {
        if (this.players.length >= 2) return false;
        this.players.push({
            id,
            name: name || `Player ${this.players.length + 1}`,
            hand: [],
            actions: { [ACTIONS.SECRET]: false, [ACTIONS.DISCARD]: false, [ACTIONS.GIFT]: false, [ACTIONS.COMPETITION]: false },
            secret: null,
            discard: [],
            side: {}, // geishaId -> [cards]
            ready: false
        });
        // Initialize geisha sides for player
        this.geishas.forEach(g => {
            this.players[this.players.length - 1].side[g.id] = [];
        });

        if (this.players.length === 2) {
            this.startRound();
        }
        return true;
    }

    startRound() {
        this.deck = [];
        GEISHAS.forEach(g => {
            for (let i = 0; i < g.value; i++) {
                this.deck.push({ geishaId: g.id, val: g.value, color: g.color });
            }
        });
        this.shuffle(this.deck);
        this.removedCard = this.deck.pop();

        this.players.forEach(p => {
            p.hand = [];
            p.actions = { [ACTIONS.SECRET]: false, [ACTIONS.DISCARD]: false, [ACTIONS.GIFT]: false, [ACTIONS.COMPETITION]: false };
            p.secret = null;
            p.discard = [];
            // Sides accumulate? No, sides are cleared each round, favor stays.
            // Actually, cards on side are cleared. Favor markers stay.
            this.geishas.forEach(g => {
                p.side[g.id] = [];
            });
            // Deal 6
            for (let i = 0; i < 6; i++) p.hand.push(this.deck.pop());
        });

        this.turnPlayerIdx = (this.round % 2 === 0) ? 1 : 0; // Winner of previous? Or alternate? Rules: Loser of previous or alternate. Let's alternate start.
        // Actually rules say: "The player who went second in the previous round becomes the starting player." -> effectively alternating.

        this.phase = 'draw';
        this.log(`Round ${this.round} started.`);
        this.nextTurn();
    }

    nextTurn() {
        // Check if round end
        const allActionsUsed = this.players.every(p => Object.values(p.actions).every(used => used));
        if (allActionsUsed) {
            this.scoreRound();
            return;
        }

        // If current player has no actions left (shouldn't happen if we alternate properly, but logic check)
        // Actually, players take turns. 4 turns each.

        // Draw card
        if (this.deck.length > 0) {
            const card = this.deck.pop();
            this.players[this.turnPlayerIdx].hand.push(card);
            this.log(`${this.players[this.turnPlayerIdx].name} drew a card.`);
        }

        this.phase = 'action';
        this.notify();

        if (this.isPvE && this.players[this.turnPlayerIdx].id === 'AI') {
            setTimeout(() => this.aiMove(), 1000);
        }
    }

    shuffle(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    }

    handleAction(playerId, actionData) {
        const pIdx = this.players.findIndex(p => p.id === playerId);
        if (pIdx !== this.turnPlayerIdx && this.phase !== 'wait_response') return { error: "Not your turn" };

        const player = this.players[pIdx];
        const { type, cards, piles, choice } = actionData;

        if (this.phase === 'action') {
            if (player.actions[type]) return { error: "Action already used" };

            // Validate cards are in hand
            // ... (omitted for brevity, assume valid for now or add checks)

            if (type === ACTIONS.SECRET) {
                // cards: [index]
                const card = player.hand.splice(cards[0], 1)[0];
                player.secret = card;
                player.actions[type] = true;
                this.log(`${player.name} used Secret.`);
                this.endTurn();
            } else if (type === ACTIONS.DISCARD) {
                // cards: [index, index]
                // Sort indices desc to splice correctly
                cards.sort((a, b) => b - a);
                cards.forEach(idx => player.discard.push(player.hand.splice(idx, 1)[0]));
                player.actions[type] = true;
                this.log(`${player.name} used Discard.`);
                this.endTurn();
            } else if (type === ACTIONS.GIFT) {
                // cards: [index, index, index]
                const selected = cards.map(idx => player.hand[idx]);
                // Remove from hand
                cards.sort((a, b) => b - a);
                cards.forEach(idx => player.hand.splice(idx, 1));

                this.pendingAction = { type, sourcePlayerIdx: pIdx, cards: selected };
                this.phase = 'wait_response';
                this.log(`${player.name} offers a Gift.`);
                player.actions[type] = true;
                this.notify();

                if (this.isPvE && this.players[1 - pIdx].id === 'AI') {
                    setTimeout(() => this.aiResponse(), 1000);
                }
            } else if (type === ACTIONS.COMPETITION) {
                // piles: [[idx, idx], [idx, idx]]
                const pile1 = piles[0].map(idx => player.hand[idx]);
                const pile2 = piles[1].map(idx => player.hand[idx]);

                // Remove from hand
                const allIndices = [...piles[0], ...piles[1]].sort((a, b) => b - a);
                allIndices.forEach(idx => player.hand.splice(idx, 1));

                this.pendingAction = { type, sourcePlayerIdx: pIdx, piles: [pile1, pile2] };
                this.phase = 'wait_response';
                this.log(`${player.name} proposes a Competition.`);
                player.actions[type] = true;
                this.notify();

                if (this.isPvE && this.players[1 - pIdx].id === 'AI') {
                    setTimeout(() => this.aiResponse(), 1000);
                }
            }
        } else if (this.phase === 'wait_response') {
            // Opponent choosing
            if (pIdx === this.pendingAction.sourcePlayerIdx) return { error: "Waiting for opponent" };

            const sourcePlayer = this.players[this.pendingAction.sourcePlayerIdx];

            if (this.pendingAction.type === ACTIONS.GIFT) {
                // choice: index of card in pendingAction.cards
                const chosenCard = this.pendingAction.cards.splice(choice, 1)[0];
                // Player gets chosen
                player.side[chosenCard.geishaId].push(chosenCard);
                // Source gets remaining
                this.pendingAction.cards.forEach(c => sourcePlayer.side[c.geishaId].push(c));

                this.log(`${player.name} chose a gift.`);
                this.pendingAction = null;
                this.endTurn();
            } else if (this.pendingAction.type === ACTIONS.COMPETITION) {
                // choice: 0 or 1 (pile index)
                const chosenPile = this.pendingAction.piles[choice];
                const otherPile = this.pendingAction.piles[1 - choice];

                chosenPile.forEach(c => player.side[c.geishaId].push(c));
                otherPile.forEach(c => sourcePlayer.side[c.geishaId].push(c));

                this.log(`${player.name} chose a pile.`);
                this.pendingAction = null;
                this.endTurn();
            }
        }
        return { success: true };
    }

    endTurn() {
        this.turnPlayerIdx = 1 - this.turnPlayerIdx;
        this.nextTurn();
    }

    scoreRound() {
        this.log("Scoring round...");
        // Reveal secrets
        this.players.forEach(p => {
            if (p.secret) {
                p.side[p.secret.geishaId].push(p.secret);
                const gName = this.geishas.find(g => g.id === p.secret.geishaId).name;
                this.log(`${p.name} revealed secret: ${p.secret.val} (${gName})`);
                p.secret = null;
            }
        });

        // Calculate favor
        this.geishas.forEach(g => {
            const p0Count = this.players[0].side[g.id].length;
            const p1Count = this.players[1].side[g.id].length;

            if (p0Count > p1Count) g.favor = 0;
            else if (p1Count > p0Count) g.favor = 1;
            // If tie, favor remains
        });

        // Check Win
        const p0Score = this.calculateScore(0);
        const p1Score = this.calculateScore(1);

        this.log(`Score - ${this.players[0].name}: ${p0Score.geishas} Geishas, ${p0Score.points} Points`);
        this.log(`Score - ${this.players[1].name}: ${p1Score.geishas} Geishas, ${p1Score.points} Points`);

        let winner = null;
        // 11 points rule
        if (p0Score.points >= 11 && p1Score.points < 11) winner = 0;
        else if (p1Score.points >= 11 && p0Score.points < 11) winner = 1;
        else if (p0Score.points >= 11 && p1Score.points >= 11) {
            if (p0Score.points > p1Score.points) winner = 0;
            else if (p1Score.points > p0Score.points) winner = 1;
            else winner = 'draw'; // or check geishas
        }

        // 4 Geishas rule (if no points winner yet)
        if (winner === null) {
            if (p0Score.geishas >= 4 && p1Score.geishas < 4) winner = 0;
            else if (p1Score.geishas >= 4 && p0Score.geishas < 4) winner = 1;
            else if (p0Score.geishas >= 4 && p1Score.geishas >= 4) {
                if (p0Score.points > p1Score.points) winner = 0;
                else if (p1Score.points > p0Score.points) winner = 1;
            }
        }

        if (winner !== null) {
            this.winner = winner;
            this.phase = 'game_end';
            this.log(`Game Over! Winner: ${winner === 'draw' ? 'Draw' : this.players[winner].name}`);
        } else {
            this.round++;
            this.startRound();
        }
        this.notify();
    }

    calculateScore(playerIdx) {
        let geishas = 0;
        let points = 0;
        this.geishas.forEach(g => {
            if (g.favor === playerIdx) {
                geishas++;
                points += g.value;
            }
        });
        return { geishas, points };
    }

    log(msg) {
        this.logs.push(`[${new Date().toLocaleTimeString()}] ${msg}`);
        if (this.logs.length > 50) this.logs.shift();
    }

    notify() {
        this.lastUpdateTime = Date.now();
    }

    // --- AI Implementation ---
    async aiMove() {
        console.log('AI Move Triggered. Turn:', this.turnPlayerIdx);

        // Double check it's AI turn
        if (this.players[this.turnPlayerIdx].id !== 'AI') return;

        try {
            // Small delay for realism
            await new Promise(resolve => setTimeout(resolve, 1000));

            const ai = this.players.find(p => p.id === 'AI');
            const availableActions = Object.entries(ai.actions)
                .filter(([_, used]) => !used)
                .map(([type]) => type);
            if (availableActions.length === 0) return;

            const actionType = availableActions[Math.floor(Math.random() * availableActions.length)];

            // Ensure we have enough cards for the action (sanity check)
            // Secret: 1, Discard: 2, Gift: 3, Comp: 4
            // In standard rules, this is always true if we follow the flow, but let's be safe.

            if (actionType === ACTIONS.SECRET) {
                this.handleAction('AI', { type: actionType, cards: [0] });
            } else if (actionType === ACTIONS.DISCARD) {
                this.handleAction('AI', { type: actionType, cards: [0, 1] });
            } else if (actionType === ACTIONS.GIFT) {
                this.handleAction('AI', { type: actionType, cards: [0, 1, 2] });
            } else if (actionType === ACTIONS.COMPETITION) {
                if (ai.hand.length < 4) {
                    console.error("AI Error: Not enough cards for Competition", ai.hand.length);
                    // Try to pick another action if possible, or just return to avoid crash
                    return;
                }
                this.handleAction('AI', { type: actionType, piles: [[0, 1], [2, 3]] });
            }
        } catch (e) {
            console.error("AI Move Error:", e);
        }
    }

    aiResponse() {
        try {
            if (this.phase !== 'wait_response') return;
            // AI chooses
            if (this.pendingAction.type === ACTIONS.GIFT) {
                this.handleAction('AI', { choice: 0 });
            } else if (this.pendingAction.type === ACTIONS.COMPETITION) {
                this.handleAction('AI', { choice: 0 });
            }
        } catch (e) {
            console.error("AI Response Error:", e);
        }
    }
}

// --- Server State ---
const games = {};

// --- HTTP Server ---
const server = http.createServer((req, res) => {
    const parsedUrl = url.parse(req.url, true);
    const { pathname, query } = parsedUrl;

    // CORS
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') {
        res.writeHead(204);
        res.end();
        return;
    }

    if (pathname === '/') {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end(HTML_CONTENT);
    } else if (pathname === '/create') {
        const gameId = crypto.randomBytes(4).toString('hex');
        const isPvE = query.mode === 'pve';
        const game = new HanamikojiGame(gameId, isPvE);
        game.addPlayer(query.playerId || 'P1', 'Player 1');
        if (isPvE) {
            game.addPlayer('AI', 'AI Bot');
        }
        games[gameId] = game;
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ gameId }));
    } else if (pathname === '/join') {
        const { gameId, playerId } = query;
        const game = games[gameId];
        if (!game) {
            res.writeHead(404);
            res.end(JSON.stringify({ error: 'Game not found' }));
            return;
        }
        if (game.addPlayer(playerId || 'P2', 'Player 2')) {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ success: true }));
        } else {
            // Check if player is already in
            const p = game.players.find(p => p.id === playerId);
            if (p) {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ success: true, reconnected: true }));
            } else {
                res.writeHead(400);
                res.end(JSON.stringify({ error: 'Game full' }));
            }
        }
    } else if (pathname === '/action') {
        let body = '';
        req.on('data', chunk => body += chunk);
        req.on('end', () => {
            try {
                const data = JSON.parse(body);
                const game = games[data.gameId];
                if (game) {
                    const result = game.handleAction(data.playerId, data.action);
                    res.writeHead(200, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify(result));
                } else {
                    res.writeHead(404);
                    res.end(JSON.stringify({ error: 'Game not found' }));
                }
            } catch (e) {
                res.writeHead(500);
                res.end(JSON.stringify({ error: e.message }));
            }
        });
    } else if (pathname === '/events') {
        const { gameId, playerId } = query;
        const game = games[gameId];
        if (!game) {
            res.writeHead(404);
            res.end();
            return;
        }

        res.writeHead(200, {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        });

        const sendState = () => {
            // Filter state for player (hide opponent hand, etc)
            const pIdx = game.players.findIndex(p => p.id === playerId);
            const player = game.players[pIdx];
            const opponent = game.players[1 - pIdx];

            const state = {
                id: game.id,
                round: game.round,
                phase: game.phase,
                turnPlayerIdx: game.turnPlayerIdx,
                pendingAction: game.pendingAction, // Should filter if secret?
                logs: game.logs,
                geishas: game.geishas,
                me: player ? {
                    ...player,
                    index: pIdx,
                    hand: player.hand // Reveal own hand
                } : null,
                opponent: opponent ? {
                    ...opponent,
                    hand: opponent.hand.length, // Hide opponent hand count only
                    secret: opponent.secret ? true : false // Hide secret value
                } : null,
                winner: game.winner
            };
            res.write(`data: ${JSON.stringify(state)}\n\n`);
        };

        const interval = setInterval(() => {
            // Simple polling for changes or just push every X ms
            // In a real app, we'd use an event emitter. Here we poll the lastUpdateTime.
            // For simplicity, just push every 500ms.
            sendState();
        }, 500);

        req.on('close', () => {
            clearInterval(interval);
        });
    } else if (pathname.startsWith('/data/images/')) {
        const safePath = path.normalize(pathname).replace(/^(\.\.[\/\\])+/, '');
        const filePath = path.join(__dirname, safePath);
        fs.readFile(filePath, (err, data) => {
            if (err) {
                res.writeHead(404);
                res.end('Image not found');
            } else {
                res.writeHead(200, { 'Content-Type': 'image/jpeg' });
                res.end(data);
            }
        });
    } else {
        res.writeHead(404);
        res.end();
    }
});

server.listen(PORT, () => {
    console.log(`Hanamikoji Server running at http://localhost:${PORT}`);
});

// --- Frontend Content ---
const HTML_CONTENT = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hanamikoji Online</title>
    <script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #1a1a2e;
            --bg-panel: #16213e;
            --accent: #e94560;
            --text: #ecf0f1;
            --card-w: 70px;
            --card-h: 105px;
            --geisha-w: 90px;
            --geisha-h: 140px;
        }
        body { margin: 0; background: var(--bg-dark); color: var(--text); font-family: 'Noto Sans JP', sans-serif; overflow: hidden; user-select: none; }
        #app { height: 100vh; display: flex; flex-direction: column; position: relative; z-index: 1; }
        
        /* Sakura Background */
        #sakura-container { position: fixed; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 0; overflow: hidden; }
        .sakura {
            position: absolute;
            background: #ffb7c5;
            border-radius: 100% 0 100% 0;
            opacity: 0.6;
            animation: fall linear infinite, sway ease-in-out infinite alternate;
        }
        @keyframes fall { to { top: 110%; } }
        @keyframes sway { from { transform: translateX(0) rotate(0deg); } to { transform: translateX(100px) rotate(360deg); } }

        .lobby { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; background: radial-gradient(circle at center, #2c3e50, #000); position: relative; z-index: 2; }
        .title { font-size: 4rem; margin-bottom: 2rem; color: var(--accent); text-shadow: 0 0 20px rgba(233,69,96,0.5); letter-spacing: 5px; }
        .menu { display: flex; flex-direction: column; gap: 1rem; width: 300px; }
        input { padding: 15px; border-radius: 5px; border: none; background: rgba(255,255,255,0.1); color: white; font-size: 1.2rem; text-align: center; }
        button { padding: 15px; border-radius: 5px; border: none; background: var(--accent); color: white; font-size: 1.2rem; cursor: pointer; transition: all 0.2s; font-weight: bold; }
        button:hover { transform: scale(1.05); box-shadow: 0 0 15px var(--accent); }
        
        /* Layout Update: 3 Columns */
        .game-container { flex: 1; display: flex; flex-direction: row; position: relative; overflow: hidden; z-index: 2; }
        
        .side-panel {
            width: 220px;
            background: rgba(0,0,0,0.2);
            border-right: 1px solid rgba(255,255,255,0.1);
            display: flex; flex-direction: column;
            padding: 10px;
            box-sizing: border-box;
            overflow-y: auto;
        }
        .side-panel.right { border-left: 1px solid rgba(255,255,255,0.1); border-right: none; }
        
        .panel-section { margin-bottom: 20px; }
        .panel-title { font-size: 1rem; color: #aaa; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid #444; padding-bottom: 5px; }
        
        .game-board { flex: 1; display: flex; flex-direction: column; position: relative; padding: 10px; }

        .top-bar { display: flex; justify-content: space-between; align-items: center; padding: 0 20px; background: rgba(0,0,0,0.3); height: 40px; margin-bottom: 5px; border-radius: 5px; }
        
        .opponent-area { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; border-bottom: 1px solid rgba(255,255,255,0.1); }
        .player-area { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; border-top: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px; }
        .center-area { flex: 1.5; display: flex; justify-content: center; align-items: center; gap: 10px; padding: 10px 0; }

        .card {
            width: var(--card-w); height: var(--card-h);
            border-radius: 6px;
            background: #fff;
            display: flex; justify-content: center; align-items: center;
            font-weight: bold; font-size: 1.5rem; color: #333;
            cursor: pointer;
            transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
            position: relative;
            box-shadow: 0 2px 5px rgba(0,0,0,0.5);
            border: 2px solid #444;
            background-image: linear-gradient(135deg, rgba(255,255,255,0.4) 0%, rgba(255,255,255,0) 100%);
        }
        .card:hover { transform: translateY(-8px) scale(1.05); box-shadow: 0 10px 20px rgba(0,0,0,0.4); z-index: 10; }
        .card.selected { transform: translateY(-15px) scale(1.05); border-color: var(--accent); box-shadow: 0 0 15px var(--accent); z-index: 10; }
        .card.opponent { background: #34495e; background-image: repeating-linear-gradient(45deg, #2c3e50 25%, transparent 25%, transparent 75%, #2c3e50 75%, #2c3e50); background-size: 10px 10px; }
        
        .card.mini { width: 50px; height: 75px; font-size: 1rem; cursor: default; }
        .card.mini:hover { transform: none; }

        .geisha {
            width: var(--geisha-w); height: var(--geisha-h);
            background: #333;
            border-radius: 8px;
            position: relative;
            display: flex; flex-direction: column; align-items: center;
            border: 2px solid #555;
            transition: all 0.3s;
            /* overflow: hidden; Removed to allow markers outside */
        }
        .geisha-img { flex: 1; width: 100%; border-radius: 6px 6px 0 0; background-size: cover; background-position: top center; opacity: 1; }
        .geisha-val { height: 25px; width: 100%; background: rgba(0,0,0,0.6); color: white; display: flex; justify-content: center; align-items: center; font-weight: bold; border-radius: 0 0 6px 6px; }
        
        .favor-marker {
            width: 24px; height: 24px; border-radius: 50%;
            background: radial-gradient(circle, #ffd700, #b8860b);
            position: absolute; left: 50%; transform: translateX(-50%);
            box-shadow: 0 2px 4px rgba(0,0,0,0.8);
            transition: top 0.5s cubic-bezier(0.68, -0.55, 0.27, 1.55);
            z-index: 20;
            border: 2px solid #fff;
        }
        
        .side-cards { position: absolute; width: 100%; display: flex; justify-content: center; gap: 2px; }
        .side-cards.top { top: -40px; }
        .side-cards.bottom { bottom: -40px; }
        .mini-card { width: 20px; height: 30px; border-radius: 2px; border: 1px solid #000; }

        .actions-row { display: flex; gap: 15px; margin: 10px 0; }
        .action-token {
            width: 50px; height: 50px; border-radius: 50%;
            background: #2c3e50; border: 2px solid #7f8c8d;
            display: flex; justify-content: center; align-items: center;
            font-size: 0.7rem; color: #bdc3c7;
            cursor: pointer; text-transform: uppercase; font-weight: bold;
            transition: all 0.2s;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .action-token:hover:not(.used) { transform: scale(1.1); border-color: var(--accent); color: white; }
        .action-token.used { opacity: 0.4; filter: grayscale(1); cursor: default; transform: scale(0.9); }
        .action-token.active { background: var(--accent); border-color: white; color: white; box-shadow: 0 0 15px var(--accent); }

        .hand { display: flex; gap: 10px; padding: 10px; min-height: 110px; }
        
        .overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); z-index: 100; display: flex; justify-content: center; align-items: center; backdrop-filter: blur(5px); }
        .modal { background: var(--bg-panel); padding: 30px; border-radius: 15px; border: 1px solid var(--accent); max-width: 600px; width: 90%; text-align: center; box-shadow: 0 0 30px rgba(0,0,0,0.5); }
        .modal h2 { margin-top: 0; color: var(--accent); }
        
        /* Right Panel Choices */
        .choice-container { display: flex; flex-direction: column; gap: 15px; margin-top: 10px; }
        .pile { padding: 10px; border: 2px dashed #555; border-radius: 10px; cursor: pointer; transition: 0.2s; background: rgba(255,255,255,0.05); }
        .pile:hover { border-color: var(--accent); background: rgba(233,69,96,0.1); transform: scale(1.02); }
        
        .toast { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); background: var(--accent); color: white; padding: 10px 20px; border-radius: 20px; font-weight: bold; z-index: 200; animation: popIn 0.3s; }
        @keyframes popIn { from { transform: translate(-50%, 20px); opacity: 0; } to { transform: translate(-50%, 0); opacity: 1; } }
    </style>
</head>
<body>
    <div id="sakura-container"></div>
    <div id="app">
        <!-- Lobby -->
        <div v-if="!game" class="lobby">
            <div class="title">花見小路</div>
            <div class="menu">
                <input v-model="playerName" placeholder="Enter Your Name" maxlength="10">
                <button @click="createGame('pvp')">Create Online Game</button>
                <button @click="createGame('pve')">Play vs AI</button>
                <div style="display:flex; gap:5px;">
                    <input v-model="joinId" placeholder="Game ID" style="flex:1;">
                    <button @click="joinGame" style="flex:0 0 80px;">Join</button>
                </div>
            </div>
        </div>

        <!-- Game -->
        <div v-else class="game-container">
            <!-- Left Panel: My Secret & Discard -->
            <div class="side-panel left">
                <div class="panel-section">
                    <div class="panel-title">My Secret</div>
                    <div v-if="game.me && game.me.secret" class="card mini" :style="{ background: game.me.secret.color }">
                        <div class="card-inner">{{ game.me.secret.val }}</div>
                    </div>
                    <div v-else style="color:#666; font-style:italic; font-size:0.9rem;">None</div>
                </div>
                
                <div class="panel-section">
                    <div class="panel-title">My Discard</div>
                    <div style="display:flex; gap:5px; flex-wrap:wrap;">
                        <div v-for="(c, i) in (game.me ? game.me.discard : [])" :key="i" class="card mini" :style="{ background: c.color }">
                            <div class="card-inner">{{ c.val }}</div>
                        </div>
                        <div v-if="!game.me || game.me.discard.length === 0" style="color:#666; font-style:italic; font-size:0.9rem;">None</div>
                    </div>
                </div>
            </div>

            <!-- Center Game Board -->
            <div class="game-board">
                <!-- Header -->
                <div class="top-bar">
                    <div>Room: {{ game.id }}</div>
                    <div>{{ opponent ? opponent.name : 'Waiting for opponent...' }}</div>
                </div>

                <!-- Opponent Area -->
                <div class="opponent-area">
                    <div class="hand" style="transform: scale(0.7);">
                        <div v-for="n in (opponent ? opponent.hand : 0)" :key="n" class="card opponent"></div>
                    </div>
                    <div class="actions-row">
                        <div v-for="(used, action) in (opponent ? opponent.actions : {})" :key="action" 
                             class="action-token" :class="{ used: used }">
                            {{ action[0] }}
                        </div>
                    </div>
                </div>

                <!-- Center Board (Geishas) -->
                <div class="center-area">
                    <div v-for="g in game.geishas" :key="g.id" class="geisha" :style="{ borderColor: g.color }">
                        <!-- Opponent Side Cards -->
                        <div class="side-cards top">
                            <div v-for="c in getOpponentSide(g.id)" :key="c.val+c.color" class="mini-card" :style="{ background: c.color }"></div>
                        </div>
                        
                        <!-- Geisha Image/Color -->
                        <div class="geisha-img" :style="{ backgroundImage: 'url(' + g.image + ')', backgroundColor: g.color }"></div>
                        <div class="geisha-val">{{ g.value }}</div>
                        
                        <!-- Favor Marker -->
                        <div class="favor-marker" :style="{ top: getFavorPos(g.favor) }"></div>

                        <!-- Player Side Cards -->
                        <div class="side-cards bottom">
                            <div v-for="c in getMySide(g.id)" :key="c.val+c.color" class="mini-card" :style="{ background: c.color }"></div>
                        </div>
                    </div>
                </div>

                <!-- Player Area -->
                <div class="player-area">
                    <!-- Actions -->
                    <div class="actions-row">
                        <div v-for="(used, action) in (game.me ? game.me.actions : {})" :key="action" 
                             class="action-token" 
                             :class="{ used: used, active: selectedAction === action }"
                             @click="selectAction(action)">
                            {{ action }}
                        </div>
                    </div>
                    
                    <!-- Hand -->
                    <div class="hand">
                        <div v-for="(card, idx) in (game.me ? game.me.hand : [])" :key="idx" 
                             class="card" 
                             :class="{ selected: selectedCards.includes(idx) }"
                             :style="{ background: card.color }"
                             @click="toggleCard(idx)">
                            <div class="card-inner">
                                <span style="font-size: 2rem; color: rgba(0,0,0,0.5);">{{ card.val }}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 10px; height: 30px; color: #aaa; font-size: 1.2rem; font-weight: bold;">
                        {{ statusMessage }}
                    </div>
                    
                    <button v-if="canConfirm" @click="confirmAction" style="padding: 10px 40px; font-size: 1.2rem; margin-top: 10px;">Confirm</button>
                </div>
            </div>

            <!-- Right Panel: Interaction -->
            <div class="side-panel right">
                <div class="panel-section">
                    <div class="panel-title">Action</div>
                    
                    <div v-if="waitingForResponse">
                        <div style="color: #e94560; font-weight: bold; margin-bottom: 10px;">Waiting for Opponent...</div>
                        <div style="color: #aaa; font-size: 0.9rem;">The opponent is choosing from the piles you offered.</div>
                    </div>

                    <div v-if="mustRespond">
                        <div style="color: #e94560; font-weight: bold; margin-bottom: 10px;">{{ respondTitle }}</div>
                        <div class="choice-container">
                            <div v-for="(option, idx) in responseOptions" :key="idx" class="pile" @click="submitResponse(idx)">
                                <div style="display:flex; gap:5px; justify-content:center;">
                                    <div v-for="c in option" class="card mini" :style="{ background: c.color }">
                                        <div class="card-inner">{{ c.val }}</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div v-if="!waitingForResponse && !mustRespond" style="color:#666; font-style:italic; font-size:0.9rem;">
                        No pending actions.
                    </div>
                </div>
            </div>

            <!-- Overlays -->
            <div v-if="game.winner !== null" class="overlay">
                <div class="modal">
                    <h1>{{ game.winner === 'draw' ? 'Draw!' : (isWinner ? 'You Win!' : 'You Lose') }}</h1>
                    <button @click="game = null">Back to Menu</button>
                </div>
            </div>
            
            <div v-if="toast" class="toast">{{ toast }}</div>
        </div>
    </div>

    <script>
        const { createApp, ref, computed, watch, onMounted } = Vue;

        // Sakura Effect
        const createSakura = () => {
            const container = document.getElementById('sakura-container');
            if (!container) return;
            const el = document.createElement('div');
            el.className = 'sakura';
            el.style.left = Math.random() * 100 + '%';
            el.style.top = '-10px';
            const size = Math.random() * 10 + 10;
            el.style.width = size + 'px';
            el.style.height = size + 'px';
            el.style.animationDuration = (Math.random() * 3 + 4) + 's, ' + (Math.random() * 2 + 2) + 's';
            el.style.animationDelay = '0s, ' + (Math.random() * 1) + 's';
            container.appendChild(el);
            setTimeout(() => el.remove(), 8000);
        };
        setInterval(createSakura, 300);

        createApp({
            setup() {
                const playerName = ref('Player');
                const joinId = ref('');
                const game = ref(null);
                const myId = ref(localStorage.getItem('hana_pid') || 'P' + Math.floor(Math.random()*10000));
                localStorage.setItem('hana_pid', myId.value);
                
                const selectedCards = ref([]); // indices
                const selectedAction = ref(null);
                const toast = ref(null);

                // Computed
                const opponent = computed(() => game.value ? game.value.opponent : null);
                const isMyTurn = computed(() => game.value && game.value.turnPlayerIdx === game.value.me.index);
                
                const connect = (gameId) => {
                    const evtSource = new EventSource(\`/events?gameId=\${gameId}&playerId=\${myId.value}\`);
                    evtSource.onmessage = (e) => {
                        const data = JSON.parse(e.data);
                        game.value = data;
                        if (data.winner !== null) evtSource.close();
                    };
                    evtSource.onerror = () => {
                        // showToast("Connection lost");
                        evtSource.close();
                    };
                };

                const createGame = async (mode) => {
                    const res = await fetch(\`/create?playerId=\${myId.value}&mode=\${mode}\`, { method: 'POST' });
                    const data = await res.json();
                    connect(data.gameId);
                };

                const joinGame = async () => {
                    if (!joinId.value) return;
                    const res = await fetch(\`/join?gameId=\${joinId.value}&playerId=\${myId.value}\`, { method: 'POST' });
                    const data = await res.json();
                    if (data.error) showToast(data.error);
                    else connect(joinId.value);
                };

                const toggleCard = (idx) => {
                    if (selectedCards.value.includes(idx)) {
                        selectedCards.value = selectedCards.value.filter(i => i !== idx);
                    } else {
                        selectedCards.value.push(idx);
                    }
                };

                const selectAction = (action) => {
                    if (game.value.me.actions[action]) return;
                    selectedAction.value = action;
                    selectedCards.value = [];
                };

                const canConfirm = computed(() => {
                    if (!selectedAction.value) return false;
                    const count = selectedCards.value.length;
                    if (selectedAction.value === 'secret' && count === 1) return true;
                    if (selectedAction.value === 'discard' && count === 2) return true;
                    if (selectedAction.value === 'gift' && count === 3) return true;
                    if (selectedAction.value === 'competition' && count === 4) return true;
                    return false;
                });

                const confirmAction = async () => {
                    if (!canConfirm.value) return;
                    
                    const payload = {
                        gameId: game.value.id,
                        playerId: myId.value,
                        action: {
                            type: selectedAction.value,
                            cards: [...selectedCards.value]
                        }
                    };

                    if (selectedAction.value === 'competition') {
                        payload.action.piles = [
                            [selectedCards.value[0], selectedCards.value[1]],
                            [selectedCards.value[2], selectedCards.value[3]]
                        ];
                        delete payload.action.cards;
                    }

                    const res = await fetch('/action', {
                        method: 'POST',
                        body: JSON.stringify(payload)
                    });
                    const data = await res.json();
                    if (data.error) showToast(data.error);
                    else {
                        selectedAction.value = null;
                        selectedCards.value = [];
                    }
                };

                const mustRespond = computed(() => {
                    if (!game.value || !game.value.pendingAction) return false;
                    return game.value.me.index !== game.value.pendingAction.sourcePlayerIdx; 
                });

                const waitingForResponse = computed(() => {
                    if (!game.value || !game.value.pendingAction) return false;
                    return game.value.me.index === game.value.pendingAction.sourcePlayerIdx;
                });

                const respondTitle = computed(() => {
                    if (!game.value.pendingAction) return '';
                    return game.value.pendingAction.type === 'gift' ? 'Choose a Gift' : 'Choose a Pile';
                });

                const responseOptions = computed(() => {
                    if (!game.value.pendingAction) return [];
                    if (game.value.pendingAction.type === 'gift') {
                        return game.value.pendingAction.cards.map(c => [c]);
                    } else {
                        return game.value.pendingAction.piles;
                    }
                });

                const submitResponse = async (choiceIdx) => {
                    const res = await fetch('/action', {
                        method: 'POST',
                        body: JSON.stringify({
                            gameId: game.value.id,
                            playerId: myId.value,
                            action: { choice: choiceIdx }
                        })
                    });
                    const data = await res.json();
                    if (data.error) showToast(data.error);
                };

                const showToast = (msg) => {
                    toast.value = msg;
                    setTimeout(() => toast.value = null, 3000);
                };

                const statusMessage = computed(() => {
                    if (!game.value) return '';
                    if (waitingForResponse.value) return 'Waiting for opponent to choose...';
                    if (mustRespond.value) return 'Your turn to choose!';
                    if (game.value.turnPlayerIdx === game.value.me.index) return 'Your Turn';
                    return 'Opponent Turn';
                });

                const getFavorPos = (favor) => {
                    if (favor === null) return '50%';
                    // Position outside the card
                    if (favor === game.value.me.index) return 'calc(100% + 5px)'; 
                    return '-25px';
                };

                const getMySide = (gid) => {
                    if (!game.value || !game.value.me) return [];
                    return game.value.me.side[gid] || [];
                };
                const getOpponentSide = (gid) => {
                    if (!game.value || !game.value.opponent) return [];
                    return game.value.opponent.side[gid] || [];
                };
                
                const isWinner = computed(() => {
                    if (!game.value || game.value.winner === null) return false;
                    return game.value.winner === game.value.me.index;
                });

                return {
                    playerName, joinId, game, myId,
                    createGame, joinGame,
                    selectedCards, selectedAction, toggleCard, selectAction,
                    confirmAction, canConfirm,
                    mustRespond, waitingForResponse, respondTitle, responseOptions, submitResponse,
                    opponent, statusMessage, toast,
                    getFavorPos, getMySide, getOpponentSide, isWinner
                };
            }
        }).mount('#app');
    </script>
</body>
</html>
`;
