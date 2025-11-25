# Hanamikoji Online (花見小路)

A beautiful, real-time online implementation of the classic board game **Hanamikoji**.

![Game Interface](screenshot.png)

## Overview

Hanamikoji is a 2-player strategy card game where players compete to win the favor of 7 Geishas by collecting their preferred items. This web version brings the elegant mechanics of the original game to your browser with a modern, polished interface.

## Features

-   **Online Multiplayer**: Play against friends in real-time.
-   **AI Mode**: Sharpen your skills against an AI opponent.
-   **Beautiful Visuals**:
    -   High-quality anime-style Geisha artwork.
    -   Atmospheric Sakura (cherry blossom) falling effect.
    -   Smooth card animations and interactions.
-   **Intuitive UI**:
    -   3-column layout for clear information hierarchy.
    -   Side panels for tracking secret/discarded cards and managing actions.
    -   Clear visual indicators for Geisha favor.

## How to Run

1.  **Prerequisites**: Ensure you have [Node.js](https://nodejs.org/) installed.
2.  **Install Dependencies**:
    ```bash
    npm install
    ```
    *(Note: This project uses standard Node.js modules, so `npm install` might not be strictly necessary if no external packages are used, but good practice.)*
3.  **Start the Server**:
    ```bash
    node hanamikoji_full.js
    ```
4.  **Play**: Open your browser and navigate to `http://localhost:3000`.

## Game Rules

1.  **Goal**: Win 4 Geishas OR score 11 points.
2.  **Turn Structure**:
    -   Draw a card.
    -   Perform one of 4 actions (Secret, Discard, Gift, Competition).
    -   Each action can be used only once per round.
3.  **Scoring**: At the end of the round (after 4 turns each), favor is calculated based on who has more item cards for each Geisha.

## Technologies

-   **Backend**: Node.js (HTTP, EventSource for real-time updates)
-   **Frontend**: Vue.js 3 (CDN), CSS3 (Flexbox, Animations)
