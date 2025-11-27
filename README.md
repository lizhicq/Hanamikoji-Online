# Hanamikoji Online / 花见小路在线版

A web-based implementation of the popular board game **Hanamikoji** (花见小路). This project provides a full-featured online game experience with both multiplayer (PvP) and single-player (PvE) modes.

这是一个基于 Web 的知名桌游 **花见小路 (Hanamikoji)** 的实现。本项目提供功能完整的在线游戏体验，支持多人对战 (PvP) 和单人 (PvE) 模式。

## Features / 功能特性

- **Single-File Architecture**: The entire application (Server, Game Logic, Frontend) is contained within `hanamikoji_full.js` for easy deployment.
  **单文件架构**: 整个应用（服务器、游戏逻辑、前端）都包含在 `hanamikoji_full.js` 中，便于部署。

- **Game Modes / 游戏模式**:
  - **PvP (Online)**: Create a room and invite a friend to play by sharing the Game ID.
    **PvP (在线对战)**: 创建房间并通过分享游戏 ID 邀请好友对战。
  - **PvE (vs AI)**: Challenge an AI opponent with adjustable difficulty.
    **PvE (人机对战)**: 挑战 AI 对手，支持多种难度调节。

- **AI Difficulty Levels / AI 难度等级**:
  - **Normal**: Random moves. (普通：随机行动)
  - **Hard**: Heuristic-based strategy. (困难：基于启发式策略)
  - **Expert**: Monte Carlo Tree Search (MCTS) for high-level play. (专家：基于蒙特卡洛树搜索 MCTS)

- **Modern Frontend / 现代前端**:
  - Built with **Vue.js 3**. (基于 Vue.js 3 构建)
  - Real-time updates using **Server-Sent Events (SSE)**. (使用 SSE 实现实时更新)
  - Beautiful UI with animations and Sakura background effects. (精美的 UI，包含动画和樱花背景特效)

## Prerequisites / 前置要求

- **Node.js** (v14 or higher recommended / 推荐 v14 或更高版本)

## Installation & Usage / 安装与使用

1. **Clone the repository / 克隆仓库**
   ```bash
   git clone <repository-url>
   cd Hanamikoji2
   ```

2. **Run the Server / 运行服务器**
   Since the project uses only Node.js built-in modules (`http`, `fs`, `crypto`, etc.), no `npm install` is strictly required for the main script.
   由于项目仅使用 Node.js 内置模块，无需安装额外依赖。

   ```bash
   node hanamikoji_full.js
   ```

3. **Play the Game / 开始游戏**
   Open your browser and visit:
   打开浏览器并访问：
   `http://localhost:3000`

## Project Structure / 项目结构

- `hanamikoji_full.js`: The main entry point containing server code, game logic, and frontend HTML/JS.
  主入口文件，包含服务器代码、游戏逻辑和前端 HTML/JS。
- `data/`: Contains static assets like images.
  包含图片等静态资源。
