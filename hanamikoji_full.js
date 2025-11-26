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
        this.winner = null;
        this.logs = [];
        this.roundReport = null;
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

        if (type === 'next_round') {
            if (this.phase === 'round_end') {
                this.nextRound();
                return { success: true };
            }
            return { error: "Cannot advance round now" };
        }

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

        // Prepare Report Data BEFORE clearing secrets/discards (though we need to reveal secrets first for favor calc)
        // Actually, we reveal secrets to side, but we should keep a record of what they were for the report.
        const p0 = this.players[0];
        const p1 = this.players[1];

        const report = {
            p0: { name: p0.name, secret: p0.secret, discard: [...p0.discard] },
            p1: { name: p1.name, secret: p1.secret, discard: [...p1.discard] }
        };

        // Reveal secrets to side for calculation
        this.players.forEach(p => {
            if (p.secret) {
                p.side[p.secret.geishaId].push(p.secret);
                const gName = this.geishas.find(g => g.id === p.secret.geishaId).name;
                this.log(`${p.name} revealed secret: ${p.secret.val} (${gName})`);
                // p.secret = null; // Keep it for a moment? No, logic needs it cleared or handled. 
                // We stored it in report, so we can clear it or keep it. 
                // Let's keep it in 'secret' property but maybe mark it revealed? 
                // Actually, standard logic pushes to side. We can set p.secret = null.
                p.secret = null;
            }
        });

        // Calculate favor
        this.geishas.forEach(g => {
            const p0Count = this.players[0].side[g.id].length;
            const p1Count = this.players[1].side[g.id].length;

            if (p0Count > p1Count) g.favor = 0;
            else if (p1Count > p0Count) g.favor = 1;
        });

        // Check Win
        const p0Score = this.calculateScore(0);
        const p1Score = this.calculateScore(1);

        this.log(`Score - ${p0.name}: ${p0Score.geishas} Geishas, ${p0Score.points} Points`);
        this.log(`Score - ${p1.name}: ${p1Score.geishas} Geishas, ${p1Score.points} Points`);

        report.scores = {
            p0: p0Score,
            p1: p1Score
        };
        report.geishas = JSON.parse(JSON.stringify(this.geishas)); // Snapshot
        this.roundReport = report;

        let winner = null;
        // 11 points rule
        if (p0Score.points >= 11 && p1Score.points < 11) winner = 0;
        else if (p1Score.points >= 11 && p0Score.points < 11) winner = 1;
        else if (p0Score.points >= 11 && p1Score.points >= 11) {
            if (p0Score.points > p1Score.points) winner = 0;
            else if (p1Score.points > p0Score.points) winner = 1;
            else winner = 'draw';
        }

        // 4 Geishas rule
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
            this.phase = 'round_end';
            this.log("Round ended. Waiting for next round...");
        }
        this.notify();
    }

    nextRound() {
        this.round++;
        this.roundReport = null;
        this.startRound();
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
        // Double check it's AI turn
        if (this.players[this.turnPlayerIdx].id !== 'AI') return;

        try {
            // Small delay for realism
            await new Promise(resolve => setTimeout(resolve, 1000));

            const ai = this.players.find(p => p.id === 'AI');
            const difficulty = this.difficulty || 'normal';

            if (difficulty === 'expert') {
                this.aiMoveMCTS(ai);
            } else if (difficulty === 'hard') {
                this.aiMoveSmart(ai);
            } else {
                this.aiMoveRandom(ai);
            }
        } catch (e) {
            console.error("AI Move Error:", e);
            try { this.aiMoveRandom(this.players.find(p => p.id === 'AI')); } catch (e2) { }
        }
    }

    aiResponse() {
        try {
            if (this.phase !== 'wait_response') return;
            const ai = this.players.find(p => p.id === 'AI');
            const difficulty = this.difficulty || 'normal';

            if (difficulty === 'expert') {
                this.aiResponseMCTS(ai);
            } else if (difficulty === 'hard') {
                this.aiResponseSmart(ai);
            } else {
                // Random choice
                if (this.pendingAction.type === ACTIONS.GIFT || this.pendingAction.type === ACTIONS.COMPETITION) {
                    this.handleAction('AI', { choice: Math.random() < 0.5 ? 0 : 1 });
                }
            }
        } catch (e) {
            console.error("AI Response Error:", e);
        }
    }

    // --- Random AI ---
    aiMoveRandom(ai) {
        const availableActions = Object.entries(ai.actions).filter(([_, used]) => !used).map(([type]) => type);
        if (availableActions.length === 0) return;
        const actionType = availableActions[Math.floor(Math.random() * availableActions.length)];

        if (actionType === ACTIONS.SECRET) {
            this.handleAction('AI', { type: actionType, cards: [0] });
        } else if (actionType === ACTIONS.DISCARD) {
            this.handleAction('AI', { type: actionType, cards: [0, 1] });
        } else if (actionType === ACTIONS.GIFT) {
            this.handleAction('AI', { type: actionType, cards: [0, 1, 2] });
        } else if (actionType === ACTIONS.COMPETITION) {
            this.handleAction('AI', { type: actionType, piles: [[0, 1], [2, 3]] });
        }
    }

    // --- Smart AI (Hard Mode) ---
    aiMoveSmart(ai) {
        const availableActions = Object.entries(ai.actions).filter(([_, used]) => !used).map(([type]) => type);
        if (availableActions.length === 0) return;

        // Priority: Secret > Discard > Competition > Gift (Heuristic order)
        // But we should evaluate best move for each available action.

        // For simplicity in this iteration, we pick an action based on a fixed priority if available, 
        // but optimize the CARDS chosen for that action.
        // A better approach: Evaluate all valid moves and pick the one with highest expected score.

        let bestMove = null;
        let bestScore = -Infinity;

        // We will simulate each possible action
        for (const actionType of availableActions) {
            const moves = this.generateMoves(ai, actionType);
            for (const move of moves) {
                const score = this.evaluateMove(ai, actionType, move);
                if (score > bestScore) {
                    bestScore = score;
                    bestMove = { type: actionType, ...move };
                }
            }
        }

        if (bestMove) {
            this.handleAction('AI', bestMove);
        } else {
            this.aiMoveRandom(ai);
        }
    }

    aiResponseSmart(ai) {
        // Evaluate taking pile 0 vs pile 1
        const piles = this.pendingAction.piles;

        // Simulate taking pile 0
        const score0 = this.evaluateStateAfterGain(ai, piles[0]);
        // Simulate taking pile 1
        const score1 = this.evaluateStateAfterGain(ai, piles[1]);

        // Choose the one that gives higher score
        this.handleAction('AI', { choice: score0 >= score1 ? 0 : 1 });
    }

    // --- Expert AI (MCTS / Determinization) ---
    aiMoveMCTS(ai) {
        const availableActions = Object.entries(ai.actions).filter(([_, used]) => !used).map(([type]) => type);
        if (availableActions.length === 0) return;

        // 1. Generate all possible moves
        let allMoves = [];
        for (const actionType of availableActions) {
            const moves = this.generateMoves(ai, actionType);
            moves.forEach(m => allMoves.push({ type: actionType, ...m }));
        }

        // 2. Run Simulations
        // For each move, we run K simulations.
        // In each simulation, we determinize the world (shuffle unknown cards) and play it out.
        const K = 20; // Simulations per move. Adjust for performance.
        const scores = allMoves.map(() => 0);

        for (let i = 0; i < allMoves.length; i++) {
            const move = allMoves[i];
            for (let k = 0; k < K; k++) {
                // Clone and Determinize
                const simGame = this.cloneAndDeterminize(ai.id);

                // Apply the move
                const aiIdx = simGame.players.findIndex(p => p.id === 'AI');
                simGame.handleAction('AI', move);

                // Rollout until round end
                this.rollout(simGame);

                // Score
                const p0Score = simGame.calculateScore(0);
                const p1Score = simGame.calculateScore(1);
                // AI is player 1 usually (index 1)
                const myScore = aiIdx === 0 ? p0Score : p1Score;
                const opScore = aiIdx === 0 ? p1Score : p0Score;

                // Simple win/loss or score diff
                // Win = 1, Draw = 0, Loss = -1
                // Or score diff: (MyGeishas - OpGeishas) * 10 + (MyPoints - OpPoints)
                let val = 0;
                if (myScore.geishas >= 4 || myScore.points >= 11) val += 100; // Win condition
                if (opScore.geishas >= 4 || opScore.points >= 11) val -= 100; // Loss condition

                val += (myScore.geishas - opScore.geishas) * 10;
                val += (myScore.points - opScore.points);

                scores[i] += val;
            }
        }

        // 3. Pick best move
        let bestIdx = 0;
        let maxScore = -Infinity;
        for (let i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                bestIdx = i;
            }
        }

        this.handleAction('AI', allMoves[bestIdx]);
    }

    aiResponseMCTS(ai) {
        // Similar to aiMoveMCTS but for response (choice 0 or 1)
        const moves = [{ choice: 0 }, { choice: 1 }];
        const K = 20;
        const scores = [0, 0];

        for (let i = 0; i < moves.length; i++) {
            for (let k = 0; k < K; k++) {
                const simGame = this.cloneAndDeterminize(ai.id);
                const aiIdx = simGame.players.findIndex(p => p.id === 'AI');
                simGame.handleAction('AI', moves[i]);
                this.rollout(simGame);

                const p0Score = simGame.calculateScore(0);
                const p1Score = simGame.calculateScore(1);
                const myScore = aiIdx === 0 ? p0Score : p1Score;
                const opScore = aiIdx === 0 ? p1Score : p0Score;

                let val = 0;
                if (myScore.geishas >= 4 || myScore.points >= 11) val += 100;
                if (opScore.geishas >= 4 || opScore.points >= 11) val -= 100;
                val += (myScore.geishas - opScore.geishas) * 10;
                val += (myScore.points - opScore.points);

                scores[i] += val;
            }
        }

        this.handleAction('AI', moves[scores[0] >= scores[1] ? 0 : 1]);
    }

    cloneAndDeterminize(perspectivePlayerId) {
        // Create a new game instance
        const clone = new HanamikojiGame(this.id + '_sim', this.isPvE);

        // Copy Public State
        clone.round = this.round;
        clone.turnPlayerIdx = this.turnPlayerIdx;
        clone.phase = this.phase;
        clone.geishas = JSON.parse(JSON.stringify(this.geishas));

        // Copy Players (Basic info)
        clone.players = JSON.parse(JSON.stringify(this.players));

        // Identify Known Cards
        const me = clone.players.find(p => p.id === perspectivePlayerId);
        const op = clone.players.find(p => p.id !== perspectivePlayerId);

        // Known: My Hand, My Side, Op Side, My Discard, Op Discard, My Secret (if set), Op Secret (if revealed? No, usually hidden)
        // Unknown: Op Hand, Op Secret (if set), Deck, Removed Card

        // Collect all cards that are KNOWN to be somewhere
        const knownCards = [];

        // My Hand
        me.hand.forEach(c => knownCards.push(c));
        // My Side
        Object.values(me.side).forEach(pile => pile.forEach(c => knownCards.push(c)));
        // Op Side
        Object.values(op.side).forEach(pile => pile.forEach(c => knownCards.push(c)));
        // My Discard
        me.discard.forEach(c => knownCards.push(c));
        // Op Discard
        op.discard.forEach(c => knownCards.push(c));
        // My Secret
        if (me.secret) knownCards.push(me.secret);

        // Also pending action cards might be known
        if (this.pendingAction) {
            clone.pendingAction = JSON.parse(JSON.stringify(this.pendingAction));
            // If pending action has cards/piles, they are known (visible on table)
            if (clone.pendingAction.cards) clone.pendingAction.cards.forEach(c => knownCards.push(c));
            if (clone.pendingAction.piles) clone.pendingAction.piles.forEach(pile => pile.forEach(c => knownCards.push(c)));
        }

        // Reconstruct the "Unknown Pool"
        // Total cards = 21 (7 geishas * values)
        // Actually we can just regenerate the full deck and subtract known cards
        const fullDeck = [];
        GEISHAS.forEach(g => {
            for (let i = 0; i < g.value; i++) {
                fullDeck.push({ geishaId: g.id, val: g.value, color: g.color });
            }
        });

        // Filter out known cards
        // We need to match by value and color/geishaId.
        // Since cards are identical objects effectively, we can just count.
        const unknownPool = [];
        const knownCounts = {}; // key -> count
        knownCards.forEach(c => {
            const key = `${c.geishaId}-${c.val}`; // Use geishaId and val for uniqueness
            knownCounts[key] = (knownCounts[key] || 0) + 1;
        });

        fullDeck.forEach(c => {
            const key = `${c.geishaId}-${c.val}`;
            if (knownCounts[key] > 0) {
                knownCounts[key]--;
            } else {
                unknownPool.push(c);
            }
        });

        // Shuffle Unknown Pool
        this.shuffle(unknownPool);

        // Distribute Unknowns
        // 1. Removed Card (1)
        clone.removedCard = unknownPool.pop();

        // 2. Op Secret (if they used secret action)
        if (op.actions[ACTIONS.SECRET] && !op.secret) { // If they used it but we don't know it
            // Wait, if they used secret, it's in their 'secret' slot, but hidden from us.
            // In 'clone.players', we copied everything. If 'this.players' had the secret visible (server state), 
            // then clone has it. But 'determinize' means we should HIDE it and randomize it if we are simulating from AI perspective?
            // Actually, 'this' is the SERVER state, so it has perfect info.
            // But we want to simulate from AI's perspective (imperfect info).
            // So we must overwrite Op's secret and hand with random cards from the pool.
            op.secret = unknownPool.pop();
        } else if (op.actions[ACTIONS.SECRET] && op.secret) {
            // If we somehow know it (e.g. end of round), keep it. But usually we don't.
            // In server state, op.secret is set. We should overwrite it with a random one from pool 
            // unless we are cheating. MCTS should respect hidden info.
            // So yes, overwrite.
            // Wait, if we overwrite, we must ensure the original 'op.secret' was NOT added to 'knownCards'.
            // In my 'knownCards' logic above, I did NOT add op.secret. Correct.
            op.secret = unknownPool.pop(); // Overwrite with random from pool
        } else {
            op.secret = null; // If they haven't used secret, it's null
        }

        // 3. Op Hand
        // How many cards should Op have?
        // We can calculate based on actions used and round phase.
        // Or just trust the count from server state.
        const opHandCount = this.players.find(p => p.id !== perspectivePlayerId).hand.length;
        op.hand = [];
        for (let i = 0; i < opHandCount; i++) {
            op.hand.push(unknownPool.pop());
        }

        // 4. Deck
        clone.deck = unknownPool; // Remaining go to deck

        // Fix Pending Action references if they point to objects we just replaced?
        // No, pendingAction cards are usually "on the table", so they are known and preserved.
        // But if pendingAction was from Op, and we randomized Op's hand...
        // If Op proposed a gift, those cards are REVEALED. They are in 'knownCards'.
        // So they are NOT in 'unknownPool'. So they are preserved. Correct.

        // Mute logs for clone
        clone.log = () => { };
        clone.notify = () => { };

        return clone;
    }

    rollout(game) {
        // Play until round ends
        let moves = 0;
        while (game.phase !== 'round_end' && game.phase !== 'game_end' && moves < 20) {
            moves++;
            // Whose turn?
            const activeP = game.players[game.turnPlayerIdx];

            if (game.phase === 'action') {
                // Pick a random valid action for the active player
                // We use a simple heuristic or random policy for rollout.
                // Random is faster and often sufficient for MCTS rollouts.
                // Smart rollout is better but slower. Let's use Random for speed.
                const available = Object.entries(activeP.actions).filter(([_, used]) => !used).map(([t]) => t);
                if (available.length === 0) {
                    // Should not happen if logic is correct, but maybe end turn?
                    game.endTurn();
                    continue;
                }
                const type = available[Math.floor(Math.random() * available.length)];

                // Construct random valid payload
                let payload = { type };
                const handIndices = activeP.hand.map((_, i) => i);

                if (type === ACTIONS.SECRET) {
                    payload.cards = [handIndices[0]];
                } else if (type === ACTIONS.DISCARD) {
                    payload.cards = [handIndices[0], handIndices[1]];
                } else if (type === ACTIONS.GIFT) {
                    payload.cards = [handIndices[0], handIndices[1], handIndices[2]];
                } else if (type === ACTIONS.COMPETITION) {
                    payload.piles = [[handIndices[0], handIndices[1]], [handIndices[2], handIndices[3]]];
                }

                game.handleAction(activeP.id, payload);

            } else if (game.phase === 'wait_response') {
                // Random choice
                game.handleAction(activeP.id, { choice: Math.random() < 0.5 ? 0 : 1 });
            } else if (game.phase === 'draw') {
                game.nextTurn();
            }
        }
    }

    // Helpers for Smart AI
    generateMoves(ai, actionType) {
        const hand = ai.hand.map((c, i) => i); // indices
        const moves = [];

        // Helper to get combinations
        const getCombinations = (arr, k) => {
            if (k === 1) return arr.map(x => [x]);
            const combs = [];
            for (let i = 0; i < arr.length - k + 1; i++) {
                const head = arr.slice(i, i + 1);
                const tailcombs = getCombinations(arr.slice(i + 1), k - 1);
                for (const tail of tailcombs) {
                    combs.push(head.concat(tail));
                }
            }
            return combs;
        };

        if (actionType === ACTIONS.SECRET) {
            // Try each card as secret
            for (let i = 0; i < hand.length; i++) moves.push({ cards: [i] });
        } else if (actionType === ACTIONS.DISCARD) {
            // Try each pair
            const combs = getCombinations(hand, 2);
            for (const c of combs) moves.push({ cards: c });
        } else if (actionType === ACTIONS.GIFT) {
            // 3 cards. Then split 1 vs 2.
            const combs = getCombinations(hand, 3);
            for (const c of combs) {
                // For a set of 3 cards {a,b,c}, possible splits: {a} vs {b,c}, {b} vs {a,c}, {c} vs {a,b}
                // We just send the 3 cards to handleAction, but we need to decide which 3.
                // Wait, handleAction for GIFT takes 'cards' (3 indices). The split happens in the UI/Logic? 
                // No, standard rules: Active player chooses 3 cards, then puts them in 2 piles?
                // My implementation: handleAction('gift') takes 3 cards. Then logic splits them? 
                // Checking handleAction... 
                // Ah, for GIFT, the backend currently just stores the 3 cards. 
                // Wait, standard rule: "Offer 3 cards". Opponent picks 1. You get 2.
                // So there are no "piles" to make. Just pick 3 cards.
                moves.push({ cards: c });
            }
        } else if (actionType === ACTIONS.COMPETITION) {
            // 4 cards. Split into 2 piles of 2.
            const combs = getCombinations(hand, 4);
            for (const c of combs) {
                // c is [i1, i2, i3, i4]. Need to split into 2 pairs.
                // {i1, i2} vs {i3, i4}
                // {i1, i3} vs {i2, i4}
                // {i1, i4} vs {i2, i3}
                moves.push({ piles: [[c[0], c[1]], [c[2], c[3]]] });
                moves.push({ piles: [[c[0], c[2]], [c[1], c[3]]] });
                moves.push({ piles: [[c[0], c[3]], [c[1], c[2]]] });
            }
        }
        return moves;
    }

    evaluateMove(ai, actionType, move) {
        // This is a simplified evaluation. 
        // We want to estimate the value of the board AFTER this action + opponent response.

        let currentScore = this.calculateCurrentScore(ai);

        if (actionType === ACTIONS.SECRET) {
            // Value = potential of this card to win a Geisha later.
            // Heuristic: High value cards (5, 4) or cards where we are losing are good to keep.
            const card = ai.hand[move.cards[0]];
            return currentScore + (card.val * 0.5); // Slight bias to keep high cards
        }
        else if (actionType === ACTIONS.DISCARD) {
            // Value = negative of the cards we lose. We want to discard useless cards.
            // Useless = Geisha already won/safe, or 2s that are not needed.
            const c1 = ai.hand[move.cards[0]];
            const c2 = ai.hand[move.cards[1]];
            return currentScore - (this.cardValue(c1) + this.cardValue(c2));
        }
        else if (actionType === ACTIONS.GIFT) {
            // We offer 3 cards. Opponent picks 1. We get 2.
            // We want to offer cards such that:
            // Opponent takes a card they don't really need (or low value), and we get 2 good cards.
            // OR Opponent takes a card they need, but we get 2 cards we need more.
            // Simulation:
            // For the 3 cards offered:
            //   Case 1: Opponent takes C1. We get C2, C3.
            //   Case 2: Opponent takes C2. We get C1, C3.
            //   Case 3: Opponent takes C3. We get C1, C2.
            // Assume opponent minimizes OUR gain (minimax).
            // So Score = min( Value(We get C2,C3), Value(We get C1,C3), Value(We get C1,C2) )
            // Actually opponent maximizes THEIR gain.
            // Let's assume Opponent Gain ~= Our Loss. Zero sum approximation.

            const cards = move.cards.map(i => ai.hand[i]);
            const v1 = this.cardValue(cards[0]);
            const v2 = this.cardValue(cards[1]);
            const v3 = this.cardValue(cards[2]);

            // Opponent will take the max value card.
            // If they take C1, we get v2+v3.
            // If they take C2, we get v1+v3.
            // If they take C3, we get v1+v2.

            // We want to maximize the outcome.
            // But we don't control what they take. They control it.
            // They will leave us with the pair that has MIN value for them? No, they take MAX value for them.
            // Let's assume Card Value is universal (good for me = good for them).

            // If they take Max(v1, v2, v3), we are left with the other two.
            // So we want to choose 3 cards such that (Sum - Max) is maximized.
            // i.e. We want to offer 3 cards that are roughly equal value, or where the "tax" is low.

            const maxV = Math.max(v1, v2, v3);
            const sum = v1 + v2 + v3;
            return currentScore + (sum - maxV);
        }
        else if (actionType === ACTIONS.COMPETITION) {
            // We offer 2 piles {A,B} vs {C,D}.
            // Opponent takes one. We get the other.
            // Opponent takes Max(Pile1, Pile2). We get Min(Pile1, Pile2).
            // We want to maximize Min(Pile1, Pile2).
            // So we should make the piles as even as possible.

            const p1 = move.piles[0].map(i => ai.hand[i]);
            const p2 = move.piles[1].map(i => ai.hand[i]);

            const val1 = this.cardValue(p1[0]) + this.cardValue(p1[1]);
            const val2 = this.cardValue(p2[0]) + this.cardValue(p2[1]);

            // Opponent takes better pile. We get worse pile.
            return currentScore + Math.min(val1, val2);
        }
        return currentScore;
    }

    evaluateStateAfterGain(ai, cardIndices) {
        // Evaluate state if AI gains these cards
        let score = 0;
        // ... (Detailed evaluation logic would go here, checking Geisha status)
        // For now, just sum card values
        // Better: Check if these cards help win a Geisha.

        // Mock implementation:
        for (const idx of cardIndices) {
            // We need to resolve the card object from the pending action piles
            // But here we just have indices relative to the pile?
            // Wait, pendingAction.piles contains actual card objects in the state sent to frontend,
            // but in backend it might be objects or indices?
            // In handleAction('competition'), we stored objects in pendingAction.piles.

            const card = idx; // In simulation, we passed card objects or values?
            // Let's assume 'cardIndices' passed from aiResponseSmart are actually card objects from the pile
            score += card.val;
        }
        return score;
    }

    cardValue(card) {
        // How valuable is this card?
        // 1. If Geisha is already won by someone, value = 0.
        // 2. If Geisha is undecided:
        //    - If we are winning, value is low (unless to secure).
        //    - If we are losing, value is high.
        //    - Higher number (5) is generally more valuable than (2).

        const g = this.geishas.find(g => g.id === card.geishaId); // Corrected from card.gid to card.geishaId
        if (g.favor !== null) return 0.1; // Already won, mostly useless (except for tie breaker in some variants, but here locked)

        // Simple heuristic: Value = Card Number (2-5)
        // Boost if it's a critical card (e.g. 5)
        return card.val;
    }

    calculateCurrentScore(ai) {
        // Estimate current standing
        return 0; // Baseline
    }

    // Fisher-Yates (Knuth) shuffle
    shuffle(array) {
        let currentIndex = array.length, randomIndex;
        while (currentIndex !== 0) {
            randomIndex = Math.floor(Math.random() * currentIndex);
            currentIndex--;
            [array[currentIndex], array[randomIndex]] = [array[randomIndex], array[currentIndex]];
        }
        return array;
    }

    restartGame() {
        this.round = 1;
        this.turnPlayerIdx = 0;
        this.winner = null;
        this.roundReport = null;
        this.logs = ['Game Restarted'];

        // Reset Geishas
        this.geishas.forEach(g => g.favor = null);

        // Reset Players
        this.players.forEach(p => {
            p.hand = [];
            p.side = {};
            this.geishas.forEach(g => p.side[g.id] = []);
            p.actions = { [ACTIONS.SECRET]: false, [ACTIONS.DISCARD]: false, [ACTIONS.GIFT]: false, [ACTIONS.COMPETITION]: false };
            p.secret = null;
            p.discard = [];
        });

        this.startRound();
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
        const difficulty = query.difficulty || 'normal';
        const game = new HanamikojiGame(gameId, isPvE);
        game.difficulty = difficulty; // Set difficulty
        game.addPlayer(query.playerId || 'P1', 'Player 1');
        if (isPvE) {
            let aiName = 'AI Bot';
            if (difficulty === 'hard') aiName = 'AI (Hard)';
            else if (difficulty === 'expert') aiName = 'AI (Expert)';
            else aiName = 'AI (Normal)';

            game.addPlayer('AI', aiName);
        }
        games[gameId] = game;
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ gameId }));
    } else if (pathname === '/join') {
        // ... (existing join logic)
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
                    if (data.action && data.action.type === 'restart') {
                        game.restartGame();
                        res.writeHead(200, { 'Content-Type': 'application/json' });
                        res.end(JSON.stringify({ success: true }));
                    } else {
                        const result = game.handleAction(data.playerId, data.action);
                        res.writeHead(200, { 'Content-Type': 'application/json' });
                        res.end(JSON.stringify(result));
                    }
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
                winner: game.winner,
                roundReport: game.roundReport
            };
            res.write(`data: ${JSON.stringify(state)}\n\n`);
        };

        const interval = setInterval(() => {
            sendState();
        }, 500);

        req.on('close', () => {
            clearInterval(interval);
        });
    }
    // Serve static files (images)
    else if (pathname.startsWith('/data/')) {
        const filePath = path.join(__dirname, pathname); // pathname includes /data/
        const ext = path.extname(filePath).toLowerCase();
        const contentTypes = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml'
        };
        const contentType = contentTypes[ext] || 'application/octet-stream';

        fs.readFile(filePath, (err, content) => {
            if (err) {
                res.writeHead(404);
                res.end('File not found');
            } else {
                res.writeHead(200, { 'Content-Type': contentType });
                res.end(content);
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
            width: 280px; /* Wider for report/log */
            background: rgba(0,0,0,0.4); /* Darker for better contrast */
            border-right: 1px solid rgba(255,255,255,0.1);
            display: flex; flex-direction: column;
            padding: 15px;
            box-sizing: border-box;
            overflow-y: auto;
            font-size: 0.9rem;
        }
        .side-panel.right { border-left: 1px solid rgba(255,255,255,0.1); border-right: none; }
        
        .panel-section { margin-bottom: 20px; background: rgba(255,255,255,0.03); padding: 10px; border-radius: 8px; }
        .panel-title { font-size: 0.9rem; color: var(--accent); margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid #444; padding-bottom: 5px; font-weight: bold; }
        
        .game-board { flex: 1; display: flex; flex-direction: column; position: relative; padding: 10px; }

        /* ... (Top Bar, Areas, Cards, Geisha styles remain mostly same, just compacting if needed) ... */
        .top-bar { display: flex; justify-content: space-between; align-items: center; padding: 0 20px; background: rgba(0,0,0,0.3); height: 30px; margin-bottom: 5px; border-radius: 5px; font-size: 0.9rem; }
        
        .opponent-area { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; border-bottom: 1px solid rgba(255,255,255,0.1); min-height: 120px; }
        .player-area { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; border-top: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px; min-height: 140px; }
        .center-area { flex: 1.5; display: flex; justify-content: center; align-items: center; gap: 8px; padding: 5px 0; }

        @keyframes deal {
            from { opacity: 0; transform: translateY(50px) scale(0.8); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }

        .card {
            width: var(--card-w); height: var(--card-h);
            border-radius: 6px;
            background: #fff;
            display: flex; justify-content: center; align-items: center;
            font-weight: bold; font-size: 1.5rem; color: #333;
            cursor: pointer;
            transition: all 0.6s cubic-bezier(0.19, 1, 0.22, 1);
            position: relative;
            box-shadow: 0 2px 5px rgba(0,0,0,0.5);
            border: 2px solid #444;
            background-image: linear-gradient(135deg, rgba(255,255,255,0.4) 0%, rgba(255,255,255,0) 100%);
            will-change: transform, box-shadow;
            animation: deal 0.6s cubic-bezier(0.19, 1, 0.22, 1) backwards;
        }
        .card:hover { transform: translateY(-12px) scale(1.1); box-shadow: 0 15px 30px rgba(0,0,0,0.4); z-index: 10; }
        .card.selected { transform: translateY(-20px) scale(1.1); border-color: var(--accent); box-shadow: 0 0 20px var(--accent); z-index: 10; }
        .card.opponent { background: #34495e; background-image: repeating-linear-gradient(45deg, #2c3e50 25%, transparent 25%, transparent 75%, #2c3e50 75%, #2c3e50); background-size: 10px 10px; }
        
        .card.mini { width: 40px; height: 60px; font-size: 0.8rem; cursor: default; border-width: 1px; }
        .card.mini:hover { transform: none; }

        .geisha {
            width: var(--geisha-w); height: var(--geisha-h);
            background: #333;
            border-radius: 8px;
            position: relative;
            display: flex; flex-direction: column; align-items: center;
            border: 2px solid #555;
            transition: all 0.3s;
        }
        .geisha-img { flex: 1; width: 100%; border-radius: 6px 6px 0 0; background-size: cover; background-position: top center; opacity: 1; }
        .geisha-val { height: 20px; width: 100%; background: rgba(0,0,0,0.6); color: white; display: flex; justify-content: center; align-items: center; font-weight: bold; border-radius: 0 0 6px 6px; font-size: 0.8rem; }
        
        .favor-marker {
            width: 20px; height: 20px; border-radius: 50%;
            background: radial-gradient(circle, #ffd700, #b8860b);
            position: absolute; left: 50%; transform: translateX(-50%);
            box-shadow: 0 2px 4px rgba(0,0,0,0.8);
            transition: top 0.6s cubic-bezier(0.19, 1, 0.22, 1);
            z-index: 20;
            border: 2px solid #fff;
        }
        
        .side-cards { position: absolute; width: 100%; display: flex; justify-content: center; gap: 1px; }
        .side-cards.top { top: -35px; }
        .side-cards.bottom { bottom: -35px; }
        .mini-card { width: 18px; height: 28px; border-radius: 2px; border: 1px solid #000; }

        .actions-row { display: flex; gap: 10px; margin: 5px 0; }
        .action-token {
            width: 40px; height: 40px; border-radius: 50%;
            background: #2c3e50; border: 2px solid #7f8c8d;
            display: flex; justify-content: center; align-items: center;
            font-size: 0.6rem; color: #bdc3c7;
            cursor: pointer; text-transform: uppercase; font-weight: bold;
            transition: all 0.2s;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .action-token:hover:not(.used) { transform: scale(1.1); border-color: var(--accent); color: white; }
        .action-token.used { opacity: 0.4; filter: grayscale(1); cursor: default; transform: scale(0.9); }
        .action-token.active { background: var(--accent); border-color: white; color: white; box-shadow: 0 0 15px var(--accent); }

        .hand { display: flex; gap: 8px; padding: 10px; min-height: 110px; }
        
        /* Log Styles */
        .log-container { flex: 1; display: flex; flex-direction: column; gap: 5px; overflow-y: auto; font-family: monospace; font-size: 0.8rem; color: #ccc; min-height: 150px; }
        .log-entry { padding: 5px; border-bottom: 1px solid rgba(255,255,255,0.05); }
        .log-entry:last-child { border-bottom: none; color: white; font-weight: bold; }

        /* Report Styles (In-Panel) */
        .report-container { display: flex; flex-direction: column; gap: 15px; }
        .report-player-block { background: rgba(0,0,0,0.2); padding: 10px; border-radius: 5px; }
        .report-label { font-size: 0.8rem; color: #aaa; margin-bottom: 5px; }
        .report-score-line { display: flex; justify-content: space-between; font-weight: bold; color: var(--accent); margin-top: 5px; }
        .report-winner-banner { background: var(--accent); color: white; padding: 10px; text-align: center; font-weight: bold; border-radius: 5px; animation: popIn 0.5s; }

        /* Right Panel Choices */
        .choice-container { display: flex; flex-direction: column; gap: 10px; margin-top: 5px; }
        .pile { padding: 8px; border: 2px dashed #555; border-radius: 8px; cursor: pointer; transition: 0.2s; background: rgba(255,255,255,0.05); }
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
                
                <div style="display:flex; gap:10px; align-items:center; justify-content:center; color:#aaa;">
                    <label>AI Difficulty:</label>
                    <select v-model="difficulty" style="padding:5px; border-radius:5px; background:rgba(255,255,255,0.1); color:white; border:none;">
                        <option value="normal">Normal</option>
                        <option value="hard">Hard (Heuristic)</option>
                        <option value="expert">Expert (MCTS)</option>
                    </select>
                </div>

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
            <!-- Left Panel: Report OR Active Info -->
            <div class="side-panel left">
                <!-- BATTLE REPORT MODE -->
                <div v-if="showReport" class="report-container">
                    <div v-if="game.phase === 'game_end'" class="report-winner-banner">
                        {{ game.winner === 'draw' ? 'Draw!' : (isWinner ? 'Victory!' : 'Defeat') }}
                    </div>
                    <div class="panel-title">Round {{ game.round }} Report</div>
                    
                    <!-- Opponent -->
                    <div class="report-player-block">
                        <div style="font-weight:bold; margin-bottom:5px;">{{ reportData.p1.name }}</div>
                        <div class="report-label">Secret</div>
                        <div v-if="reportData.p1.secret" class="card mini" :style="{ background: reportData.p1.secret.color }">
                            <div class="card-inner">{{ reportData.p1.secret.val }}</div>
                        </div>
                        <div v-else class="report-label">None</div>
                        
                        <div class="report-label" style="margin-top:5px;">Discard</div>
                        <div style="display:flex; gap:5px;">
                            <div v-for="(c, i) in reportData.p1.discard" :key="i" class="card mini" :style="{ background: c.color }">
                                <div class="card-inner">{{ c.val }}</div>
                            </div>
                        </div>
                        <div class="report-score-line">
                            <span>{{ reportData.scores.p1.geishas }} Geishas</span>
                            <span>{{ reportData.scores.p1.points }} Pts</span>
                        </div>
                    </div>

                    <!-- Me -->
                    <div class="report-player-block">
                        <div style="font-weight:bold; margin-bottom:5px;">{{ reportData.p0.name }} (You)</div>
                        <div class="report-label">Secret</div>
                        <div v-if="reportData.p0.secret" class="card mini" :style="{ background: reportData.p0.secret.color }">
                            <div class="card-inner">{{ reportData.p0.secret.val }}</div>
                        </div>
                        <div v-else class="report-label">None</div>
                        
                        <div class="report-label" style="margin-top:5px;">Discard</div>
                        <div style="display:flex; gap:5px;">
                            <div v-for="(c, i) in reportData.p0.discard" :key="i" class="card mini" :style="{ background: c.color }">
                                <div class="card-inner">{{ c.val }}</div>
                            </div>
                        </div>
                        <div class="report-score-line">
                            <span>{{ reportData.scores.p0.geishas }} Geishas</span>
                            <span>{{ reportData.scores.p0.points }} Pts</span>
                        </div>
                    </div>

                    <button v-if="game.phase === 'round_end'" @click="nextRound" style="width:100%; padding:10px;">Next Round</button>
                    <div v-if="game.phase === 'game_end'" style="display:flex; flex-direction:column; gap:10px;">
                        <button @click="playAgain" style="width:100%; padding:10px; background: #2ecc71;">Play Again</button>
                        <button @click="game = null" style="width:100%; padding:10px;">Back to Lobby</button>
                    </div>
                </div>

                <!-- ACTIVE GAME MODE -->
                <div v-else>
                    <div class="panel-section">
                        <div class="panel-title">My Secret</div>
                        <div v-if="game.me && game.me.secret" class="card mini" :style="{ background: game.me.secret.color }">
                            <div class="card-inner">{{ game.me.secret.val }}</div>
                        </div>
                        <div v-else style="color:#666; font-style:italic;">None</div>
                    </div>
                    
                    <div class="panel-section">
                        <div class="panel-title">My Discard</div>
                        <div style="display:flex; gap:5px; flex-wrap:wrap;">
                            <div v-for="(c, i) in (game.me ? game.me.discard : [])" :key="i" class="card mini" :style="{ background: c.color }">
                                <div class="card-inner">{{ c.val }}</div>
                            </div>
                            <div v-if="!game.me || game.me.discard.length === 0" style="color:#666; font-style:italic;">None</div>
                        </div>
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
                        <div v-for="(n, idx) in (opponent ? opponent.hand : 0)" :key="n" class="card opponent" :style="{ animationDelay: idx * 0.1 + 's' }"></div>
                    </div>
                    <div class="actions-row">
                        <div v-for="(used, action) in (opponent ? opponent.actions : {})" :key="action" 
                             class="action-token" :class="{ used: used }">
                            {{ getActionDisplay(action) }}
                        </div>
                    </div>
                </div>

                <!-- Center Board (Geishas) -->
                <div class="center-area">
                    <div v-for="g in game.geishas" :key="g.id" class="geisha" :style="{ borderColor: g.color }">
                        <!-- Opponent Side Cards -->
                        <div class="side-cards top">
                            <div v-for="(c, idx) in getOpponentSide(g.id)" :key="c.val+c.color" class="mini-card" :style="{ background: c.color, animationDelay: idx * 0.05 + 's' }"></div>
                        </div>
                        
                        <!-- Geisha Image/Color -->
                        <div class="geisha-img" :style="{ backgroundImage: 'url(' + g.image + ')', backgroundColor: g.color }"></div>
                        <div class="geisha-val">{{ g.value }}</div>
                        
                        <!-- Favor Marker -->
                        <div class="favor-marker" :style="{ top: getFavorPos(g.favor) }"></div>

                        <!-- Player Side Cards -->
                        <div class="side-cards bottom">
                            <div v-for="(c, idx) in getMySide(g.id)" :key="c.val+c.color" class="mini-card" :style="{ background: c.color, animationDelay: idx * 0.05 + 's' }"></div>
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
                            {{ getActionDisplay(action) }}
                        </div>
                    </div>
                    
                    <!-- Hand -->
                    <div class="hand">
                        <div v-for="(card, idx) in (game.me ? game.me.hand : [])" :key="idx" 
                             class="card" 
                             :class="{ selected: selectedCards.includes(idx) }"
                             :style="{ background: card.color, animationDelay: idx * 0.1 + 's' }"
                             @click="toggleCard(idx)">
                            <div class="card-inner">
                                <span style="font-size: 2rem; color: rgba(0,0,0,0.5);">{{ card.val }}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 5px; height: 25px; color: #aaa; font-size: 1rem; font-weight: bold;">
                        {{ statusMessage }}
                    </div>
                    
                    <button v-if="canConfirm" @click="confirmAction" style="padding: 8px 30px; font-size: 1rem; margin-top: 5px;">Confirm</button>
                </div>
            </div>

            <!-- Right Panel: Log & Interaction -->
            <div class="side-panel right">
                <!-- Game Log -->
                <div class="panel-section" style="flex: 1; display:flex; flex-direction:column; min-height:0;">
                    <div class="panel-title">Game Log</div>
                    <div class="log-container" id="game-log">
                        <div v-for="(log, i) in (game.logs || [])" :key="i" class="log-entry">
                            {{ log }}
                        </div>
                    </div>
                </div>

                <!-- Action Interaction -->
                <div class="panel-section" style="flex: 0 0 auto;">
                    <div class="panel-title">Action</div>
                    
                    <div v-if="waitingForResponse">
                        <div style="color: #e94560; font-weight: bold; margin-bottom: 5px;">Waiting...</div>
                        <div style="color: #aaa; font-size: 0.8rem;">Opponent is choosing.</div>
                    </div>

                    <div v-if="mustRespond">
                        <div style="color: #e94560; font-weight: bold; margin-bottom: 5px;">{{ respondTitle }}</div>
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
                    
                    <div v-if="!waitingForResponse && !mustRespond" style="color:#666; font-style:italic; font-size:0.8rem;">
                        No pending actions.
                    </div>
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
                const difficulty = ref('normal'); // Default difficulty
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
                    const res = await fetch(\`/create?playerId=\${myId.value}&mode=\${mode}&difficulty=\${difficulty.value}\`, { method: 'POST' });
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
                    // selectedCards.value = []; // Keep cards selected
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

                const nextRound = async () => {
                    const res = await fetch('/action', {
                        method: 'POST',
                        body: JSON.stringify({
                            gameId: game.value.id,
                            playerId: myId.value,
                            action: { type: 'next_round' }
                        })
                    });
                    const data = await res.json();
                    if (data.error) showToast(data.error);
                };

                const playAgain = async () => {
                    const res = await fetch('/action', {
                        method: 'POST',
                        body: JSON.stringify({
                            gameId: game.value.id,
                            playerId: myId.value,
                            action: { type: 'restart' }
                        })
                    });
                    const data = await res.json();
                    if (data.error) showToast(data.error);
                    else {
                        // Reconnect to event stream if needed, but usually it stays open unless closed by winner check?
                        // Actually, if winner != null, we closed the stream. So we need to reconnect.
                        connect(game.value.id);
                    }
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

                const getActionDisplay = (action) => {
                    switch(action) {
                        case 'secret': return '1';
                        case 'discard': return '2';
                        case 'gift': return '3';
                        case 'competition': return '4';
                        default: return '?';
                    }
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

                const showReport = computed(() => {
                    return game.value && (game.value.phase === 'round_end' || game.value.phase === 'game_end') && game.value.roundReport;
                });

                const reportData = computed(() => {
                    if (!game.value || !game.value.roundReport) return null;
                    const p0 = game.value.roundReport.p0;
                    const p1 = game.value.roundReport.p1;
                    const scores = game.value.roundReport.scores;
                    
                    if (game.value.me.index === 0) {
                        return { p0: p0, p1: p1, scores: { p0: scores.p0, p1: scores.p1 } };
                    } else {
                        return { p0: p1, p1: p0, scores: { p0: scores.p1, p1: scores.p0 } };
                    }
                });

                watch(() => game.value ? game.value.logs : null, () => {
                    setTimeout(() => {
                        const el = document.getElementById('game-log');
                        if (el) el.scrollTop = el.scrollHeight;
                    }, 100);
                }, { deep: true });

                return {
                    playerName, joinId, game, myId, difficulty,
                    createGame, joinGame,
                    selectedCards, selectedAction, toggleCard, selectAction,
                    confirmAction, canConfirm,
                    mustRespond, waitingForResponse, respondTitle, responseOptions, submitResponse,
                    opponent, statusMessage, toast,
                    getFavorPos, getMySide, getOpponentSide, isWinner,
                    showReport, reportData, nextRound, getActionDisplay, playAgain
                };
            }
        }).mount('#app');
    </script>
</body>
</html>
`;
