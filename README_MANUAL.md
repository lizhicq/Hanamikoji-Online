# Hanamikoji Online - User Manual

## 1. Introduction
Welcome to **Hanamikoji Online**, a web-based implementation of the popular 2-player card game. This version supports both:
- **Online PVP**: Play against a friend over the internet.
- **Local AI**: Play against a computer opponent.

## 2. How to Run the Game

### Prerequisites
- Node.js installed on your computer.

### Steps
1. **Start the Server**:
   Open your terminal in the project directory and run:
   ```bash
   node hanamikoji_full.js
   ```
   You should see: `Hanamikoji Server running at http://localhost:3000`

2. **Generate Online URL (Optional)**:
   If you want to play with a friend over the internet, you need a public URL. You can use `localtunnel`:
   ```bash
   npx localtunnel --port 3000
   ```
   This will give you a URL like `https://cold-mice-help.loca.lt`. Share this URL with your friend.
   *Note: The first time you visit a localtunnel URL, you might need to click "Click to Continue" and possibly enter your public IP address for security verification.*

3. **Access the Game**:
   - **Local Play**: Open `http://localhost:3000` in your browser.
   - **Online Play**: Open the generated `localtunnel` URL.

## 3. How to Play

### Main Menu
- **Enter Your Name**: Type your nickname.
- **Create Online Game**: Starts a new PVP lobby. You will get a **Game ID** (e.g., `game_123`). Share this ID with your friend.
- **Play vs AI**: Starts a game against the computer immediately.
- **Join Game**: Enter a Game ID provided by a friend and click "Join".

### Game Rules (Summary)
- **Goal**: Win the favor of 4 Geishas OR score 11 charm points.
- **Turn**:
  1. Draw a card.
  2. Perform one of 4 actions (Secret, Discard, Gift, Competition).
  3. Each action can only be used ONCE per round.
- **Actions**:
  - **Secret**: Save 1 card for scoring later. (Hidden from opponent)
  - **Discard**: Discard 2 cards from your hand. (Not scored)
  - **Gift**: Offer 3 cards. Opponent picks 1, you keep 2.
  - **Competition**: Offer 4 cards in two piles (2+2). Opponent picks one pile, you keep the other.

### Controls
- **Click Card**: Select/Deselect a card from your hand.
- **Click Action Token**: Select an action to perform.
- **Confirm**: Execute the action with selected cards.
- **Piles**: When responding to Gift/Competition, click the pile you want to take.

## 4. Troubleshooting
- **Game Stuck?**: Refresh the page. The server holds the state in memory, so a full server restart (`Ctrl+C` then `node hanamikoji_full.js`) resets everything.
- **Images not loading?**: Ensure the `data/images` folder exists and contains `ayame.jpg`, `botan.jpg`, etc.
- **AI not moving?**: The AI takes about 1 second to "think". If it gets stuck, check the server console for errors.

Enjoy the game!
