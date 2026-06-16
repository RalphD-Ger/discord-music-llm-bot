# discord-music-llm-bot

A Discord bot that combines a full-featured music player with a persistent LLM personality layer, deployed 24/7 on an NVIDIA Jetson Orin Nano via systemd.

---

## Overview

This project started as a music bot and grew into a multi-function assistant. The core idea: keep a lightweight, always-on service on edge hardware that handles music playback, answers questions, generates images, and maintains memory across conversations — all through Discord.

The bot runs two independent layers:

- **Music Player** — streams YouTube and plays local audio files with real-time FFmpeg audio processing
- **LLM Brain** — stateful conversations via the xAI API (Grok 4.3) with web search, persistent user memory, and conversation history

---

## Features

### Music Player
- YouTube streaming (single videos and playlists up to 680+ tracks via cookie auth) and local MP3/FLAC playback
- Queue management: add, remove, move, shuffle, loop (track or queue), paginated list view
- Real-time FFmpeg audio processing: EBU R128 loudness normalization, bass boost EQ, night mode compression, fade-in/out
- Per-track duration tracking with `!np` progress bar
- Download YouTube tracks as 320 kbps MP3 with embedded source URL tag (for play-count deduplication across local and streamed versions)
- Persistent play statistics with top-10 ranking and favourites queue
- Auto-disconnect on idle or empty channel, with automatic voice reconnect on gateway drop

### LLM Layer
- Stateful conversations (Grok 4.3) with per-user chat history and cross-session persistence
- Live context injection: current track, genre folder, time of day
- Per-user memory system (`!remember` / `!forget`) stored as JSON
- Automatic web search via xAI tool use (used for `!explain` and general chat)
- Utility commands: `!explain`, `!compare`, `!tldr`, `!tip`, `!randomfact`, `!joke`, `!darkjoke`, `!roast`, `!summary`
- Image generation via `!imagine` (grok-imagine-image-quality model)
- Daily API call cap and per-user access control

### Infrastructure
- Deployed as a systemd service on NVIDIA Jetson Orin Nano (ARM64, JetPack 6.2)
- NVMe SSD root filesystem for reliable 24/7 operation
- Rotating Discord presence showing current track or command hints
- Secrets managed via environment file, never hardcoded

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10 (async) |
| Discord | discord.py 2.x with voice support |
| LLM | xAI SDK — Grok 4.3 (stateful chat, tool use) |
| Audio | FFmpeg, yt-dlp, PyNaCl, davey |
| Search | xAI web_search tool (native) |
| Image Gen | xAI grok-imagine-image-quality |
| Fuzzy Search | rapidfuzz |
| Audio Tags | mutagen (ID3, duration, WOAS source tracking) |
| Hardware | NVIDIA Jetson Orin Nano 8GB |
| Deployment | systemd, LAN-connected, Glasfaser uplink |

---

## Architecture

```
discord-music-llm-bot/
│
├── bot.py              # Discord client, event routing, on_message dispatcher
├── config.py           # All constants (models, paths, limits, filters)
├── brain.py            # xAI SDK integration, memory, conversation persistence
├── player.py           # Track, GuildMusic, FFmpeg engine, queue logic
└── commands.py         # All !commands (music + LLM)
```

> **Note:** The current release is a single-file implementation. Module separation is planned as the next refactor step.

### Audio Pipeline

```
YouTube URL / Local File
        ↓
  yt-dlp (stream resolve, lazy per-track)
        ↓
  FFmpeg (loudnorm → EQ → fade → PCMVolumeTransformer)
        ↓
  discord.py VoiceClient → Discord Voice Channel
```

### LLM Memory Flow

```
User message
    ↓
music context + time/day injected
    ↓
xAI stateful chat (up to 50 turns)
    ↓ (on reset)
last 20 exchanges reinjected as real user/assistant messages
    ↓
persistent JSON: ralph_conversation.json / ralph_memory.json
```

---

## Setup

### Requirements

```bash
sudo apt install ffmpeg
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file (never commit this):

```
DISCORD_TOKEN=your_discord_bot_token
XAI_API_KEY=your_xai_api_key
```

### Configuration

Edit the config block at the top of `bot.py`:

```python
MUSIC_DIR        = "/path/to/your/music"
USER_NAMES       = {123456789: "yourname"}   # Discord user ID → name
ALLOWED_USER_IDS = {123456789}               # Who can use LLM features
```

### systemd Deployment

```ini
[Unit]
Description=Discord Music LLM Bot
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/home/youruser/bot
ExecStart=/usr/bin/python3 /home/youruser/bot/bot.py
Restart=always
RestartSec=10
EnvironmentFile=/home/youruser/bot/.env

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable bot
sudo systemctl start bot
journalctl -u bot -f
```

---

## Commands

### Music
| Command | Description |
|---|---|
| `!play <link\|folder>` | Stream YouTube URL/playlist or play local genre folder |
| `!randomplay` | 50 random tracks from the full local library |
| `!np` | Now playing with progress bar |
| `!list [page]` | Queue with pagination |
| `!skip` / `!stop` / `!pause` / `!resume` | Playback control |
| `!shuffle` / `!loop` / `!clear` | Queue management |
| `!volume <0-200>` | Volume (loudnorm keeps all tracks at equal perceived loudness) |
| `!bassboost` / `!nightmode` / `!fade` | Audio filters (toggle) |
| `!download [folder]` | Save current YouTube track as 320 kbps MP3 |
| `!stats` / `!history` / `!favourites` | Play statistics |
| `!search <term>` | Fuzzy search local library |

### LLM
| Command | Description |
|---|---|
| `!explain <topic>` | Short explanation with web search |
| `!compare <A> vs <B>` | Side-by-side comparison |
| `!tldr <text>` | Summarize pasted text |
| `!tip <topic>` | Practical tip on any subject |
| `!randomfact` | Random interesting fact (category-randomized) |
| `!imagine <prompt>` | Generate image via Grok |
| `!remember` / `!forget` / `!memories` | Persistent user memory |
| `!summary` | Summarize last conversation |
| `!ping` | Latency, uptime, version |

---

## Hardware

Deployed on an **NVIDIA Jetson Orin Nano 8GB** running JetPack 6.2.2 on NVMe SSD.

The Jetson handles the Discord bot process, local music library, and persistent data files. LLM inference runs in the cloud (xAI API) — the Jetson is the always-on orchestration layer, not the compute target.

---

## License

MIT
