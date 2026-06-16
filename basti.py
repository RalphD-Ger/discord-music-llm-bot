import os
import io
import re
import time
import json
import glob
import random
import asyncio
from datetime import date, datetime
from collections import defaultdict, deque

import discord
import httpx
import yt_dlp
from mutagen import File as MutagenFile
from mutagen.id3 import ID3, WOAS, ID3NoHeaderError
from rapidfuzz import fuzz, process

from xai_sdk import AsyncClient
from xai_sdk.chat import system as xai_system, user as xai_user, assistant as xai_assistant
from xai_sdk.tools import web_search as xai_web_search

# ════════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════════

# Secrets from env-file (NEVER HARDCODE!):
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN", "")
XAI_API_KEY   = os.environ.get("XAI_API_KEY", "")

# --- Brain ---------------------------------------------------------------
# "xai"    = xAI SDK (Cloud, gRPC)
# "ollama" = local model in case you wanna use RTX Card. 
BRAIN_BACKEND = "xai"

# xAI SDK
XAI_MODEL            = "grok-4.3"
XAI_REASONING_EFFORT = "low"        # low is enough, keeps cost down
XAI_IMAGE_MODEL      = "grok-imagine-image-quality" 
XAI_TIMEOUT          = 120          # time-out after 120 seconds

# Ollama (only when BRAIN_BACKEND = "ollama")
OLLAMA_URL   = "http://localhost:11434/api/chat"   
OLLAMA_MODEL = "dark-champion"    # use the name of the model you created in your ollama env.

SYSTEM_PROMPT = """You are Basti, a helpful discord bot for playing music and chatting. 
"""

# Wer darf das LLM triggern? (Discord User-IDs als int). Musik ist fuer ALLE offen.
ALLOWED_USER_IDS = {}             # put the Discord-ID of every user that is allowed to use/chat with Basti.
MAX_LLM_CALLS_PER_DAY = 50
MAX_TURNS   = 50                  # reset after 50 messages to keep cost down, change if required
MAX_STORED_EXCHANGES = 30         # Exchanges that will be saved permanently
INJECT_EXCHANGES     = 20         # Exchanges that will injected every message for context
MAX_HISTORY = 25                  # only required if using ollama local llm

# --- Music ---------------------------------------------------------------
MUSIC_DIR  = "/home/user/Music"   # <-- Put correct directory for music download
AUDIO_EXTS = (".mp3", ".flac", ".wav", ".m4a", ".ogg", ".opus")

# Audio-Filter (FFmpeg). loudnorm = EBU R128 (-16 LUFS).
RECONNECT_OPTS = "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5"
NORMAL_FILTER  = "loudnorm=I=-16:TP=-1.5:LRA=11"
NIGHT_FILTER   = "loudnorm=I=-24:TP=-2.0:LRA=7,acompressor=threshold=-25dB:ratio=4"
BASS_FILTER    = "equalizer=f=60:width_type=o:width=1.5:g=3,equalizer=f=100:width_type=o:width=1.5:g=4"
FADE_DURATION  = 4   # 4 Seconds Fade-in und Fade-out effect
 
IDLE_TIMEOUT = 600   # 600 Seconds without any music -> Auto-Disconnect

VERSION = "1.2"

STATS_FILE = "/home/user/Music/basti_stats.json"   # Counts stats for your music

# User-Mapping: Discord-ID -> Name (for User-Memory and Personality)
USER_NAMES: dict = {0: "ralph"}

# ════════════════════════════════════════════════════════════════════════
#  STATE
# ════════════════════════════════════════════════════════════════════════
 
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
 
brain_enabled = True
_llm_usage = {"date": date.today(), "count": 0}
_idle_task_started = False
_status_task = None
_start_time = time.time()   # counts uptime
 
 
def _allowed(user_id: int) -> bool:
    return user_id in ALLOWED_USER_IDS
 
 
def _under_daily_cap() -> bool:
    today = date.today()
    if _llm_usage["date"] != today:
        _llm_usage["date"] = today
        _llm_usage["count"] = 0
    if _llm_usage["count"] >= MAX_LLM_CALLS_PER_DAY:
        return False
    _llm_usage["count"] += 1
    return True
 
 
# ════════════════════════════════════════════════════════════════════════
#  BRAIN, written for XAI but you can use whatever you like
# ════════════════════════════════════════════════════════════════════════
 
_xai_client: AsyncClient | None = None
_xai_chats: dict = {}
_xai_turns = defaultdict(int)
 
 
def _get_xai_client() -> AsyncClient:
    global _xai_client
    if _xai_client is None:
        _xai_client = AsyncClient(api_key=XAI_API_KEY, timeout=XAI_TIMEOUT)
    return _xai_client


async def _one_shot_llm(system_prompt: str, user_msg: str, with_search: bool = False) -> str:
    xc = _get_xai_client()
    kwargs = dict(
        model=XAI_MODEL,
        messages=[xai_system(system_prompt)],
        reasoning_effort="low",
    )
    if with_search:
        kwargs["tools"] = [xai_web_search()]
    chat = xc.chat.create(**kwargs)
    chat.append(xai_user(user_msg))
    response = await chat.sample()
    return response.content
 
 
async def _xai_reply(user_id: int, message: str, music_context: str = "") -> str:
    xc = _get_xai_client()
    if user_id not in _xai_chats or _xai_turns[user_id] >= MAX_TURNS:
        msgs = [xai_system(_build_system_prompt(user_id))]
        for ex in _load_conversation(user_id)[-INJECT_EXCHANGES:]:
            msgs.append(xai_user(ex["user"]))
            msgs.append(xai_assistant(ex["basti"]))
        _xai_chats[user_id] = xc.chat.create(
            model=XAI_MODEL,
            messages=msgs,
            reasoning_effort=XAI_REASONING_EFFORT,
            tools=[xai_web_search()],
        )
        _xai_turns[user_id] = 0
    chat = _xai_chats[user_id]
    full_msg = f"[{music_context}]\n{message}" if music_context else message
    chat.append(xai_user(full_msg))
    response = await chat.sample()
    chat.append(response)
    _xai_turns[user_id] += 1
    return response.content
 
 
_ollama_histories = defaultdict(list)
 
 
async def _ollama_reply(user_id: int, message: str) -> str:
    history = _ollama_histories[user_id]
    history.append({"role": "user", "content": message})
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
        _ollama_histories[user_id] = history
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *history],
        "stream": False,
        "options": {"temperature": 0.85, "num_ctx": 12000},
    }
    async with httpx.AsyncClient(timeout=600.0) as http:
        r = await http.post(OLLAMA_URL, json=payload)
        r.raise_for_status()
        data = r.json()
    reply = data["message"]["content"]
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
    history.append({"role": "assistant", "content": reply})
    return reply
 
 
async def ask_brain(user_id: int, user_message: str, music_context: str = "") -> str:
    if BRAIN_BACKEND == "xai":
        return await _xai_reply(user_id, user_message, music_context)
    elif BRAIN_BACKEND == "ollama":
        return await _ollama_reply(user_id, user_message)
    raise ValueError(f"Unbekanntes BRAIN_BACKEND: {BRAIN_BACKEND}")
 
 
# ════════════════════════════════════════════════════════════════════════
#  Play-Statistics (persistent in JSON)
# ════════════════════════════════════════════════════════════════════════
 
# Structure:  { ref: {"title": str, "kind": "local"|"yt", "count": int} }
_stats: dict = {}
 
 
def _load_stats():
    global _stats
    try:
        with open(STATS_FILE, "r", encoding="utf-8") as f:
            _stats = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        _stats = {}
 
 
def _save_stats():
    try:
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(_stats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️  Stats saving failed: {e}")


def _stats_key(track) -> str:
    if track.kind == "yt":
        return track.ref
    try:
        tags = ID3(track.ref)
        woas = tags.get("WOAS")
        if woas:
            return woas.url 
    except Exception:
        pass
    return track.ref
 
 
def _record_play(track):
    key = _stats_key(track)
    rec = _stats.get(key)
    if rec:
        rec["count"] += 1
        rec["title"] = track.title
    else:
        _stats[key] = {"title": track.title, "kind": track.kind, "count": 1}
    _save_stats()
 
 
_load_stats()
 

# ════════════════════════════════════════════════════════════════════════
#  USER-MEMORY (Per User in <name>_memory.json)
# ════════════════════════════════════════════════════════════════════════

def _memory_file(user_id: int) -> str | None:
    name = USER_NAMES.get(user_id)
    return f"{name}_memory.json" if name else None
 
 
def _load_memory(user_id: int) -> list:
    path = _memory_file(user_id)
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
 
 
def _save_memory(user_id: int, facts: list):
    path = _memory_file(user_id)
    if not path:
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=2)
 
 
def _conversation_file(user_id: int) -> str | None:
    name = USER_NAMES.get(user_id)
    return f"{name}_conversation.json" if name else None
 
 
def _load_conversation(user_id: int) -> list:
    path = _conversation_file(user_id)
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
 
 
def _save_conversation(user_id: int, user_msg: str, basti_reply: str):
    path = _conversation_file(user_id)
    if not path:
        return
    history = _load_conversation(user_id)
    history.append({"user": user_msg, "basti": basti_reply})
    if len(history) > MAX_STORED_EXCHANGES:
        history = history[-MAX_STORED_EXCHANGES:]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
 
 
def _build_system_prompt(user_id: int) -> str:
    """Build System Prompt with User-Kontext and Memory"""
    name = USER_NAMES.get(user_id)
    facts = _load_memory(user_id)
    parts = [SYSTEM_PROMPT]
    if name:
        parts.append(f"You speak with {name.capitalize()}.")
    if facts:
        facts_block = "\n".join(f"- {f}" for f in facts)
        parts.append(
            f"What you know about {name.capitalize() if name else 'this User'} actually:\n{facts_block}"
        )
    return "\n\n".join(parts)
 

# ════════════════════════════════════════════════════════════════════════
#  MUSIC – Bibliothek & yt-dlp
# ════════════════════════════════════════════════════════════════════════
 
def _collect_audio(root: str) -> list:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(AUDIO_EXTS):
                out.append(os.path.join(dirpath, fn))
    return out
 
 
def _search_library(query: str, files: list, limit: int = 10) -> list:
    """Fuzzy-Search for file-names"""
    names = [os.path.basename(f) for f in files]
    stems = [os.path.splitext(n)[0] for n in names]
    results = process.extract(
        query, stems, scorer=fuzz.WRatio, processor=str.lower, limit=limit
    )
    out = []
    for _stem, score, idx in results:
        if score >= 60:
            out.append((names[idx], files[idx], score))
    return out
 
 
def _entry_url(entry: dict) -> str:
    u = entry.get("url")
    if u and u.startswith("http"):
        return u
    return f"https://www.youtube.com/watch?v={entry.get('id')}"


# Expliziter Node-Pfad damit yt-dlp ihn findet unabhaengig vom Python-PATH
YTDL_BASE          = {}   # fuer Streaming/Download — kein Cookie noetig, war vorher stabil
YTDL_BASE_PLAYLIST = {    # fuer Playlist-Extraktion — Cookie damit alle Tracks sichtbar sind
    "cookiefile": "/home/ralph/basti/cookies.txt",
}
 
 
def _ytdl_flat(url: str) -> list:
    opts = {**YTDL_BASE_PLAYLIST, "extract_flat": True, "quiet": True, "noplaylist": False, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if "entries" in info and info["entries"]:
        return [(_entry_url(e), e.get("title", "Unknown")) for e in info["entries"] if e]
    return [(url, info.get("title", "Unknown"))]
 
 
def _ytdl_resolve(url: str):
    opts = {**YTDL_BASE, "format": "bestaudio/best", "quiet": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return info["url"], info.get("title", "Unknown"), info.get("duration")
 
 
def _ytdl_download(url: str, output_dir: str) -> str:
    """Laedt Audio herunter, konvertiert zu MP3 und bettet YT-URL als Tag ein."""
    opts = {
        **YTDL_BASE,
        "format": "bestaudio/best",
        "quiet": True,
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "320",
        }],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "Unknown")
        # Exakter Pfad nach Postprocessor: Endung durch .mp3 ersetzen
        mp3_path = os.path.splitext(ydl.prepare_filename(info))[0] + ".mp3"
 
    # YT-Quell-URL als WOAS-Tag einbetten -> Stats zaehlen mit Streaming zusammen
    try:
        try:
            tags = ID3(mp3_path)
        except ID3NoHeaderError:
            tags = ID3()
            tags.save(mp3_path)
            tags = ID3(mp3_path)
        tags["WOAS"] = WOAS(url=url)
        tags.save(mp3_path)
    except Exception as e:
        print(f"⚠️  WOAS-Tag konnte nicht eingebettet werden: {e}")
 
    return f"{title}.mp3"
 
def _apply_fade(af: str, duration: float | None, fade: bool) -> str:
    """Haengt Fade-in und Fade-out an die FFmpeg-Filterchain wenn fade aktiv."""
    if not fade:
        return af
    parts = [af, f"afade=t=in:ss=0:d={FADE_DURATION}"]
    if duration and duration > FADE_DURATION * 2:
        st = duration - FADE_DURATION
        parts.append(f"afade=t=out:st={st:.2f}:d={FADE_DURATION}")
    return ",".join(parts)


class Track:
    def __init__(self, kind: str, ref: str, title: str):
        self.kind = kind
        self.ref = ref
        self.title = title
        self.duration: float | None = None
 
    async def audio_source(self, af: str, fade: bool = False) -> discord.FFmpegPCMAudio:
        if self.kind == "local":
            try:
                audio = MutagenFile(self.ref)
                self.duration = audio.info.length if audio else None
            except Exception:
                self.duration = None
            final_af = _apply_fade(af, self.duration, fade)
            return discord.FFmpegPCMAudio(self.ref, options=f"-vn -af {final_af}")
        loop = asyncio.get_running_loop()
        stream_url, title, duration = await loop.run_in_executor(None, _ytdl_resolve, self.ref)
        self.title = title
        self.duration = duration
        final_af = _apply_fade(af, self.duration, fade)
        return discord.FFmpegPCMAudio(stream_url, before_options=RECONNECT_OPTS, options=f"-vn -af {final_af}")
 
 
class GuildMusic:
    def __init__(self):
        self.queue: list = []
        self.voice = None
        self.current = None
        self.volume = 0.5       # 0.5 = 50%
        self.nightmode = False
        self.bassboost = False
        self.fade = False
        self.loop_mode = "off"     # off | track | queue
        self.skip_flag = False
        self.last_activity = time.time()
        self.last_search: list = []   # letzte !search-Treffer: [(title, path), ...]
        self.history: deque = deque(maxlen=50)  # zuletzt gespielte Tracks
        self._play_start: float = 0.0
        self._paused_at: float | None = None
        self._total_paused: float = 0.0
        self.intentional_disconnect: bool = False
        self.last_channel = None        # letzter Voice-Channel (discord.VoiceChannel)
        self.last_text_channel = None   # letzter Text-Channel fuer Reconnect-Meldung        
 
 
players: dict = {}
 
 
def get_player(guild_id: int) -> GuildMusic:
    if guild_id not in players:
        players[guild_id] = GuildMusic()
    return players[guild_id]
 
 
async def ensure_voice(message: discord.Message):
    if not message.author.voice or not message.author.voice.channel:
        await message.reply("🔇 Geh erst in einen Voice-Channel.")
        return None
    channel = message.author.voice.channel
    gm = get_player(message.guild.id)
    if gm.voice and gm.voice.is_connected():
        if gm.voice.channel != channel:
            await gm.voice.move_to(channel)
    else:
        gm.voice = await channel.connect()
    gm.last_activity = time.time()
    gm.last_channel = channel
    gm.last_text_channel = message.channel
    gm.intentional_disconnect = False
    return gm.voice
 
 
# ════════════════════════════════════════════════════════════════════════
#  MUSIK – Playback-Engine
# ════════════════════════════════════════════════════════════════════════

def _elapsed(gm) -> float:
    if gm._play_start == 0:
        return 0.0
    if gm._paused_at:
        return gm._paused_at - gm._play_start - gm._total_paused
    return time.time() - gm._play_start - gm._total_paused
 
 
def _fmt_time(s: float) -> str:
    s = max(0, int(s))
    return f"{s // 60}:{s % 60:02d}"
 
 
def _progress_bar(elapsed: float, duration: float | None, width: int = 15) -> str:
    if duration and duration > 0:
        ratio = min(elapsed / duration, 1.0)
        filled = int(ratio * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {_fmt_time(elapsed)} / {_fmt_time(duration)}"
    return f"[{'█' * width}] {_fmt_time(elapsed)} / ?"

def _build_af(gm: GuildMusic) -> str:
    """Baut die FFmpeg-Filterchain aus aktiven Toggles zusammen."""
    base = NIGHT_FILTER if gm.nightmode else NORMAL_FILTER
    if gm.bassboost:
        return BASS_FILTER + "," + base
    return base


async def _play_next(guild_id: int):
    gm = players.get(guild_id)
    if gm is None or gm.voice is None or not gm.voice.is_connected():
        return
 
    # Loop des gerade beendeten Tracks (Skip umgeht den Loop)
    finished = gm.current
    if finished is not None and not gm.skip_flag:
        if gm.loop_mode == "track":
            gm.queue.insert(0, finished)
        elif gm.loop_mode == "queue":
            gm.queue.append(finished)
    gm.skip_flag = False
 
    if not gm.queue:
        gm.current = None
        return
 
    track = gm.queue.pop(0)
    gm.current = track
    gm.last_activity = time.time()
    gm._play_start = time.time()
    gm._paused_at = None
    gm._total_paused = 0.0
    _record_play(track)
    gm.history.append(track.title)
 
    af = _build_af(gm)
    try:
        base = await track.audio_source(af, fade=gm.fade)
    except Exception as e:
        print(f"⚠️  Track übersprungen ({track.title}): {e}")
        await _play_next(guild_id)
        return
 
    source = discord.PCMVolumeTransformer(base, volume=gm.volume)
    loop = asyncio.get_running_loop()
 
    def _after(err):
        if err:
            print(f"⚠️  Playback-Fehler: {err}")
        asyncio.run_coroutine_threadsafe(_play_next(guild_id), loop)
 
    gm.voice.play(source, after=_after)
 
 
async def start_playing(guild_id: int):
    gm = players[guild_id]
    if gm.voice is None or not gm.voice.is_connected():
        return
    if gm.voice.is_playing() or gm.voice.is_paused():
        return
    await _play_next(guild_id)
 
 
async def _enqueue_from_arg(message: discord.Message, arg: str):
    """Gibt eine Liste Track-Objekte aus Link/Ordner zurueck (oder None bei Fehler)."""
    if arg.startswith(("http://", "https://")):
        loop = asyncio.get_running_loop()
        try:
            items = await loop.run_in_executor(None, _ytdl_flat, arg)
        except Exception as e:
            await message.reply(f"❌ Konnte den Link nicht lesen: {e}")
            return None
        return [Track("yt", url, title) for url, title in items]
    folder = os.path.join(MUSIC_DIR, arg)
    if not os.path.isdir(folder):
        await message.reply(f"📁 Ordner `{arg}` nicht gefunden in {MUSIC_DIR}.")
        return None
    files = _collect_audio(folder)
    if not files:
        await message.reply("Keine Audio-Files im Ordner.")
        return None
    random.shuffle(files)
    return [Track("local", f, os.path.basename(f)) for f in files]
 
 
# ════════════════════════════════════════════════════════════════════════
#  MUSIK – Commands
# ════════════════════════════════════════════════════════════════════════
 
async def cmd_play(message: discord.Message):
    arg = message.content[len("!play"):].strip()
    if not arg:
        await message.reply("Usage: `!play <youtube-link>`  oder  `!play <ordnername>`")
        return
    gm = get_player(message.guild.id)
 
    # !play <nummer> -> Treffer aus der letzten !search-Liste abspielen
    if arg.isdigit() and gm.last_search:
        idx = int(arg) - 1
        if not (0 <= idx < len(gm.last_search)):
            await message.reply("Ungültige Such-Nummer.")
            return
        title, path = gm.last_search[idx]
        if await ensure_voice(message) is None:
            return
        gm.queue.append(Track("local", path, title))
        await message.reply(f"➕ `{title}` zur Queue.")
        await start_playing(message.guild.id)
        return
 
    if await ensure_voice(message) is None:
        return
    tracks = await _enqueue_from_arg(message, arg)
    if tracks is None:
        return
    gm.queue.extend(tracks)
    await message.reply(f"➕ {len(tracks)} Track(s) zur Queue.")
    await start_playing(message.guild.id)
 
 
async def cmd_playnext(message: discord.Message):
    arg = message.content[len("!playnext"):].strip()
    if not arg:
        await message.reply("Usage: `!playnext <link/ordner>`")
        return
    if await ensure_voice(message) is None:
        return
    tracks = await _enqueue_from_arg(message, arg)
    if tracks is None:
        return
    gm = get_player(message.guild.id)
    gm.queue[0:0] = tracks      # vorne einfuegen
    await message.reply(f"⏭️ {len(tracks)} Track(s) als Nächstes eingereiht.")
    await start_playing(message.guild.id)
 
 
async def cmd_randomplay(message: discord.Message):
    if await ensure_voice(message) is None:
        return
    files = _collect_audio(MUSIC_DIR)
    if not files:
        await message.reply("Keine lokale Musik gefunden.")
        return
    random.shuffle(files)
    files = files[:50]
    gm = get_player(message.guild.id)
    for f in files:
        gm.queue.append(Track("local", f, os.path.basename(f)))
    await message.reply(f"🔀 {len(files)} zufällige Tracks geladen.")
    await start_playing(message.guild.id)
 
 
async def cmd_skip(message: discord.Message):
    gm = get_player(message.guild.id)
    if gm.voice and (gm.voice.is_playing() or gm.voice.is_paused()):
        gm.skip_flag = True
        gm.voice.stop()
        await message.reply("⏭️ Skip.")
    else:
        await message.reply("Nichts läuft gerade.")
 
 
async def cmd_pause(message: discord.Message):
    gm = get_player(message.guild.id)
    if gm.voice and gm.voice.is_playing():
        gm.voice.pause()
        gm._paused_at = time.time()
        await message.reply("⏸️ Pausiert.")
    else:
        await message.reply("Nichts läuft gerade.")
 
 
async def cmd_resume(message: discord.Message):
    gm = get_player(message.guild.id)
    if gm.voice and gm.voice.is_paused():
        gm.voice.resume()
        if gm._paused_at:
            gm._total_paused += time.time() - gm._paused_at
            gm._paused_at = None
        gm.last_activity = time.time()
        await message.reply("▶️ Weiter.")
    else:
        await message.reply("Nichts ist pausiert.")
 
 
async def cmd_stop(message: discord.Message):
    gm = get_player(message.guild.id)
    gm.intentional_disconnect = True
    gm.queue.clear()
    if gm.voice and gm.voice.is_connected():
        await gm.voice.disconnect()
    gm.voice = None
    gm.current = None
    await message.reply("⏹️ Gestoppt, Queue geleert, Channel verlassen.")
 
 
async def cmd_clear(message: discord.Message):
    gm = get_player(message.guild.id)
    n = len(gm.queue)
    gm.queue.clear()
    await message.reply(f"🗑️ Queue geleert ({n} Tracks). Aktueller Track läuft weiter.")
 
 
async def cmd_shuffle(message: discord.Message):
    gm = get_player(message.guild.id)
    if len(gm.queue) < 2:
        await message.reply("Zu wenig Tracks in der Queue zum Shufflen.")
        return
    random.shuffle(gm.queue)
    await message.reply(f"🔀 Queue mit {len(gm.queue)} Tracks geshuffelt.")
 
 
async def cmd_remove(message: discord.Message):
    arg = message.content[len("!remove"):].strip()
    gm = get_player(message.guild.id)
    if not arg:
        await message.reply("Usage: `!remove 3`  oder  `!remove 3-7`")
        return
    try:
        if "-" in arg:
            a, b = arg.split("-", 1)
            start, end = int(a), int(b)
        else:
            start = end = int(arg)
    except ValueError:
        await message.reply("Ungueltige Nummer.")
        return
    if start < 1 or end < start or start > len(gm.queue):
        await message.reply("Bereich ausserhalb der Queue.")
        return
    end = min(end, len(gm.queue))
    removed = gm.queue[start - 1:end]
    del gm.queue[start - 1:end]
    await message.reply(f"🗑️ {len(removed)} Track(s) entfernt (Position {start}-{end}).")
 
 
async def cmd_move(message: discord.Message):
    parts = message.content[len("!move"):].split()
    gm = get_player(message.guild.id)
    if len(parts) != 2:
        await message.reply("Usage: `!move <von> <nach>`")
        return
    try:
        frm, to = int(parts[0]), int(parts[1])
    except ValueError:
        await message.reply("Zahlen angeben.")
        return
    if not (1 <= frm <= len(gm.queue)) or not (1 <= to <= len(gm.queue)):
        await message.reply("Position ausserhalb der Queue.")
        return
    track = gm.queue.pop(frm - 1)
    gm.queue.insert(to - 1, track)
    await message.reply(f"↕️ `{track.title[:40]}` → Position {to}.")
 
 
async def cmd_loop(message: discord.Message):
    arg = message.content[len("!loop"):].strip().lower()
    gm = get_player(message.guild.id)
    if arg in ("", "track", "song"):
        gm.loop_mode = "off" if gm.loop_mode == "track" else "track"
    elif arg == "queue":
        gm.loop_mode = "off" if gm.loop_mode == "queue" else "queue"
    elif arg == "off":
        gm.loop_mode = "off"
    else:
        await message.reply("Usage: `!loop` (Track) · `!loop queue` · `!loop off`")
        return
    labels = {"off": "aus", "track": "🔂 Einzeltrack", "queue": "🔁 Queue"}
    await message.reply(f"Loop: {labels[gm.loop_mode]}")
 
 
async def cmd_volume(message: discord.Message):
    arg = message.content[len("!volume"):].strip()
    gm = get_player(message.guild.id)
    if not arg:
        await message.reply(f"🔊 Aktuelle Lautstärke: {int(gm.volume * 100)} %")
        return
    try:
        v = int(arg)
    except ValueError:
        await message.reply("Zahl zwischen 0 und 200 angeben.")
        return
    v = max(0, min(200, v))
    gm.volume = v / 100
    if gm.voice and isinstance(gm.voice.source, discord.PCMVolumeTransformer):
        gm.voice.source.volume = gm.volume      # live anpassen
    await message.reply(f"🔊 Lautstärke: {v} %")
 
 
async def cmd_nightmode(message: discord.Message):
    gm = get_player(message.guild.id)
    gm.nightmode = not gm.nightmode
    await message.reply(
        f"🌙 Nightmode {'an' if gm.nightmode else 'aus'} (gilt ab dem nächsten Track)."
    )


async def cmd_bassboost(message: discord.Message):
    gm = get_player(message.guild.id)
    gm.bassboost = not gm.bassboost
    await message.reply(
        f"🔊 Bassboost {'an' if gm.bassboost else 'aus'} (gilt ab dem nächsten Track)."
    )

async def cmd_fade(message: discord.Message):
    gm = get_player(message.guild.id)
    gm.fade = not gm.fade
    await message.reply(
        f"🌊 Fade {'an' if gm.fade else 'aus'} ({FADE_DURATION}s Ein- & Ausblenden, gilt ab dem naechsten Track)."
    )
 
async def cmd_history(message: discord.Message):
    gm = get_player(message.guild.id)
    if not gm.history:
        await message.reply("Noch nichts gespielt in dieser Session.")
        return
    recent = list(reversed(gm.history))[:20]   # neueste zuerst
    rows = ["⏮️  Zuletzt gespielt:", "─" * 46]
    for i, title in enumerate(recent, 1):
        rows.append(f" {i:>2}. {title[:42]}")
    out = "```\n" + "\n".join(rows) + "\n```"
    await message.reply(out)

async def cmd_summary(message: discord.Message):
    if not _allowed(message.author.id):
        return
    history = _load_conversation(message.author.id)
    if not history:
        await message.reply("Noch keine Gespraechshistorie vorhanden.")
        return
    name = USER_NAMES.get(message.author.id, "User").capitalize()
    lines = []
    for ex in history:
        lines.append(f"{name}: {ex['user']}")
        lines.append(f"Basti: {ex['basti']}")
    conversation_text = "\n".join(lines)
 
    async with message.channel.typing():
        try:
            result = await _one_shot_llm(
                "Du bist ein neutraler Assistent. Fasse Gespraeche kurz auf Deutsch zusammen — "
                "maximal 3-4 Saetze, nur die wichtigsten Punkte, sachlich.",
                f"Bitte fasse dieses Gespraech zusammen:\n\n{conversation_text}"
            )
            await message.reply(f"📋 **Zusammenfassung:**\n{result}")
        except Exception as e:
            await message.reply(f"❌ Fehler: {e}")
 
 
async def cmd_genres(message: discord.Message):
    if not os.path.isdir(MUSIC_DIR):
        await message.reply(f"MUSIC_DIR nicht gefunden: {MUSIC_DIR}")
        return
    folders = sorted(
        d for d in os.listdir(MUSIC_DIR) if os.path.isdir(os.path.join(MUSIC_DIR, d))
    )
    if not folders:
        await message.reply("Keine Genre-Ordner gefunden.")
        return
    rows = ["Verfügbare Ordner:"]
    for d in folders:
        cnt = len(_collect_audio(os.path.join(MUSIC_DIR, d)))
        rows.append(f"  {d}  ({cnt})")
    out = "```\n" + "\n".join(rows) + "\n```"
    if len(out) > 1990:
        out = out[:1987] + "…```"
    await message.reply(out)
 
 
async def cmd_download(message: discord.Message):
    if not _allowed(message.author.id):
        return
    gm = get_player(message.guild.id)
    if not gm.current:
        await message.reply("Gerade laeuft nichts.")
        return
    if gm.current.kind != "yt":
        await message.reply("📁 Track ist schon lokal gespeichert.")
        return
    arg = message.content[len("!download"):].strip()
    if arg:
        target = os.path.join(MUSIC_DIR, arg)
        os.makedirs(target, exist_ok=True)
    else:
        target = MUSIC_DIR
    title = gm.current.title
    ref = gm.current.ref
    await message.reply(f"⬇️ Lade `{title}` herunter…")
    loop = asyncio.get_running_loop()
    try:
        filename = await loop.run_in_executor(None, _ytdl_download, ref, target)
        await message.reply(f"✅ Gespeichert: `{filename}` → `{target}`")
    except Exception as e:
        await message.reply(f"❌ Download fehlgeschlagen: {e}")
 
 
async def cmd_search(message: discord.Message):
    query = message.content[len("!search"):].strip()
    if not query:
        await message.reply("Usage: `!search <suchbegriff>`")
        return
    files = _collect_audio(MUSIC_DIR)
    if not files:
        await message.reply("Keine lokale Musik gefunden.")
        return
    loop = asyncio.get_running_loop()
    matches = await loop.run_in_executor(None, _search_library, query, files)
    if not matches:
        await message.reply(f"🔎 Nichts gefunden fuer `{query}`.")
        return
    gm = get_player(message.guild.id)
    gm.last_search = [(title, path) for title, path, _ in matches]
    rows = [f'🔎 Treffer fuer "{query}":', "─" * 46]
    for i, (title, _, _score) in enumerate(matches, 1):
        rows.append(f" {i:>2}. {title[:42]}")
    rows.append("─" * 46)
    rows.append("   Abspielen mit  !play <nummer>")
    out = "```\n" + "\n".join(rows) + "\n```"
    await message.reply(out)
 
 
async def cmd_stats(message: discord.Message):
    if not _stats:
        await message.reply("Noch keine Wiedergabe-Statistik vorhanden.")
        return
    ranked = sorted(_stats.values(), key=lambda r: r["count"], reverse=True)[:10]
    rows = ["🏆 Top 10 — meistgespielt", "─" * 46]
    for i, r in enumerate(ranked, 1):
        rows.append(f" {i:>2}. {r['title'][:38]:<38} {r['count']:>3}×")
    out = "```\n" + "\n".join(rows) + "\n```"
    await message.reply(out)
 
 
async def cmd_favourites(message: discord.Message):
    if not _stats:
        await message.reply("Noch keine Statistik vorhanden.")
        return
    if await ensure_voice(message) is None:
        return
    ranked = sorted(_stats.values(), key=lambda r: r["count"], reverse=True)
    gm = get_player(message.guild.id)
    added = 0
    for r in ranked:
        if r["kind"] == "local" and os.path.isfile(r["ref"]):
            gm.queue.append(Track("local", r["ref"], r["title"]))
            added += 1
            if added >= 20:
                break
    if added == 0:
        await message.reply("Keine lokalen Favoriten gefunden.")
        return
    await message.reply(f"⭐ {added} Lieblings-Tracks geladen.")
    await start_playing(message.guild.id)
 
 
async def cmd_np(message: discord.Message):
    gm = get_player(message.guild.id)
    if not gm.current:
        await message.reply("Gerade läuft nichts.")
        return
    src = "📁 lokal" if gm.current.kind == "local" else "▶️ YouTube"
    state = "⏸️ pausiert" if (gm.voice and gm.voice.is_paused()) else "▶️ läuft"
    loop_lbl = {"off": "aus", "track": "Einzeltrack", "queue": "Queue"}[gm.loop_mode]
    extra = ""
    if gm.nightmode:
        extra += " · 🌙 Nightmode"
    if gm.bassboost:
        extra += " · 🎸 Bassboost"
    if gm.fade:
        extra += " · 🌊 Fade"
    bar = _progress_bar(_elapsed(gm), gm.current.duration)
    await message.reply(
        f"**{gm.current.title}**\n"
        f"`{bar}`\n"
        f"{src} · {state} · 🔊 {int(gm.volume * 100)} % · 🔁 {loop_lbl}{extra}"
    )
 
 
async def cmd_list(message: discord.Message):
    gm = get_player(message.guild.id)
    parts = message.content.split()
    page = int(parts[1]) if (len(parts) > 1 and parts[1].isdigit()) else 1
    page = max(1, page)
 
    if not gm.current and not gm.queue:
        await message.reply("Queue ist leer.")
        return
 
    per_page = 20
    total = len(gm.queue)
    pages = max(1, (total + per_page - 1) // per_page)
    page = min(page, pages)
    start = (page - 1) * per_page
    show = gm.queue[start:start + per_page]
 
    W = 52
    rows = []
    rows.append(f"▶  {gm.current.title[:W - 3]}" if gm.current else "▶  (nichts)")
    rows.append("─" * W)
    if total == 0:
        rows.append("   (keine weiteren Tracks)")
    else:
        for idx, t in enumerate(show, start=start + 1):
            rows.append(f" {idx:>3}.  {t.title[:W - 8]}")
    rows.append("─" * W)
    rows.append(f"   {total} Track(s) · Seite {page}/{pages}")
 
    out = "```\n" + "\n".join(rows) + "\n```"
    if len(out) > 1990:
        out = out[:1987] + "…```"
    await message.reply(out)
 
 
HELP_TEXT_1 = """🎵 **Basti — Befehle — Music Bot**
 
**Wiedergabe**
!play <link|ordner>       · YouTube-Link, Playlist oder lokal abspielen
!playnext <link|ordner>   · Track/Ornder direkt als Nächstes einreihen
!randomplay               · 50 zufällige Tracks aus der lokalen Bibliothek
!pause / !resume          · Wiedergabe pausieren / fortsetzen
!skip                     · Aktuellen Track überspringen
!stop                     · Wiedergabe stoppen & Channel verlassen
 
**Queue**
!list [seite]             · Queue anzeigen (20 Tracks pro Seite)
!np                       · Aktueller Track mit Status
!shuffle                  · Queue zufällig mischen
!remove <n|n-m>           · Track(s) entfernen
!move <von> <nach>        · Track in der Queue verschieben
!clear                    · Queue leeren
!loop / !loop queue       · Track / Queue Loopen
!loop off                 · Loop ausschalten
 
**Sound**
!volume <0-200>           · Lautstärke (alle Tracks auf gleichem Pegel)
!nightmode                · Leiser + Kompression (ruhige Umgebung)
!bassboost                · Mehr Bass (60 Hz + 100 Hz Boost)
!fade                     · 4s Fade-in & Fade-out pro Track
 
**Bibliothek**
!genres / !folders      · Verfügbare Genre-Ordner anzeigen
!search <begriff>         · Lokale Fuzzy-Suche → `!play <nr>`
!stats                    · Top 10 meistgespielte Tracks
!history                  · Zuletzt gespielte Tracks
!favourites               · Top-Favoriten in die Queue laden
!download [ordner]        · Aktuellen YT-Track als MP3 speichern
 
**System**
!help                     · Diese Übersicht
!ping                     · Latenz, Uptime und Version
"""

HELP_TEXT_2 = """🎵 **Basti — Befehle — LLM Brain**

**Wissen & Info**
!explain <thema>          · Erklärt eine Thema
!compare <A> vs <B>       · Vergleicht A und B miteinander
!randomfact               · Zufälliger Fakt
!tip <thema>              · Gibt einen nützlichen Tipp zum Thema
!tldr <text>              · Fasst einen Text kurz zusammen

**Spaß**
!joke                     · Erzählt einen random Witz
!darkjoke                 · Erzählt einen bösen Witz
!roast [@user]            · Basti wird einen User roasten
!imagine <prompt>         · Erstellt mit dem Prompt ein Bild

**Memory**
!remember <fakt>          · Fakt über dich speichern
!memories                 · Gespeicherte Erinnerungen anzeigen
!forget <nr>              · Memory-Eintrag löschen
!summary                  · Zusammenfassung des letzten Gesprächs

**System**
!help                     · Diese Übersicht
!ping                     · Latenz, Uptime und Version
"""


async def cmd_ping(message: discord.Message):
    latency = round(client.latency * 1000)
    uptime_s = int(time.time() - _start_time)
    h, rem = divmod(uptime_s, 3600)
    m, s = divmod(rem, 60)
    uptime_str = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"
    await message.reply(
        f"🏓 Pong!\n"
        f"Latenz: **{latency} ms**\n"
        f"Uptime: **{uptime_str}**\n"
        f"Version: **{VERSION}**"
    )
 
 
async def cmd_randomfact(message: discord.Message):
    categories = [
        "Biologie", "Physik", "Geschichte", "Mathematik", "Astronomie",
        "Chemie", "Geographie", "Psychologie", "Technologie", "Meeresbiologie",
        "Archaeologie", "Meteorologie", "Botanik", "Genetik", "Quantenmechanik",
        "Linguistik", "Oekologie", "Medizin", "Informatik", "Philosophie",
    ]
    category = random.choice(categories)
    async with message.channel.typing():
        try:
            result = await _one_shot_llm(
                "Gib einen einzelnen zufaelligen interessanten Fakt auf Deutsch. "
                "Nur der Fakt selbst, kein Kommentar davor oder danach, 1-2 Saetze.",
                f"Zufaelliger interessanter Fakt aus dem Bereich: {category}"
            )
            await message.reply(f"💡 {result}")
        except Exception as e:
            await message.reply(f"❌ Fehler: {e}")
 
 
async def cmd_explain(message: discord.Message):
    topic = message.content[len("!explain"):].strip()
    if not topic:
        await message.reply("Usage: `!explain <thema>`")
        return
    async with message.channel.typing():
        try:
            xc = _get_xai_client()
            chat = xc.chat.create(
                model=XAI_MODEL,
                messages=[xai_system(
                    "Erklaere das folgende Thema kurz und praezise auf Deutsch. "
                    "Maximal 3 Saetze, sachlich und korrekt, keine Ausschweifungen."
                )],
                reasoning_effort="low",
                tools=[xai_web_search()],
            )
            chat.append(xai_user(topic))
            response = await chat.sample()
            result = response.content
 
            # Quellen anhaengen wenn Web Search verwendet wurde
            citations = getattr(response, "citations", None)
            if citations:
                urls = []
                for c in citations[:3]:
                    url = getattr(c, "url", None) or str(c)
                    urls.append(f"• {url}")
                result += "\n\n🌐 **Quellen:**\n" + "\n".join(urls)
 
            await send_chunked(message, f"📖 **{topic}**\n{result}")
        except Exception as e:
            await message.reply(f"❌ Fehler: {e}")
 

 
async def cmd_compare(message: discord.Message):
    arg = message.content[len("!compare"):].strip()
    if not arg or " vs " not in arg.lower():
        await message.reply("Usage: `!compare <A> vs <B>`")
        return
    async with message.channel.typing():
        try:
            result = await _one_shot_llm(
                "Vergleiche die zwei genannten Dinge sachlich auf Deutsch. "
                "Struktur: 1-2 Saetze Einordnung, dann je 2-3 Stichpunkte Vorteile pro Seite, "
                "dann ein kurzes Fazit wann man was waehlt. Kompakt halten.",
                arg,
                with_search=True
            )
            await send_chunked(message, f"⚖️ **{arg}**\n{result}")
        except Exception as e:
            await message.reply(f"❌ Fehler: {e}")
 
 
async def cmd_tip(message: discord.Message):
    topic = message.content[len("!tip"):].strip()
    if not topic:
        topic = "Allgemeinwissen oder Alltag"
    async with message.channel.typing():
        try:
            result = await _one_shot_llm(
                "Gib einen einzelnen konkreten, praktischen und wenig bekannten Tipp "
                "zum genannten Thema auf Deutsch. Actionable, kein Allgemeinplatz, "
                "1-3 Saetze. Nur der Tipp selbst.",
                f"Thema: {topic}"
            )
            await message.reply(f"💡 **Tipp ({topic}):**\n{result}")
        except Exception as e:
            await message.reply(f"❌ Fehler: {e}")

 
async def cmd_joke(message: discord.Message):
    async with message.channel.typing():
        try:
            result = await _one_shot_llm(
                "Erfinde einen originalen deutschen Dad-Joke oder Kalauer mit Wortspiel. "
                "WICHTIG: Kein uebersetzter englischer Witz — nur originales deutsches Wortspiel. "
                "Format: Frage, dann Antwort/Pointe. Kurz und groan-worthy.",
                "Erzaehl mir einen deutschen Dad-Joke."
            )
            await message.reply(f"😄 {result}")
        except Exception as e:
            await message.reply(f"❌ Fehler: {e}")
 
 
async def cmd_darkjoke(message: discord.Message):
    async with message.channel.typing():
        try:
            result = await _one_shot_llm(
                "Erzaehl einen richtig brutalen schwarzen Humor Witz auf Deutsch. "
                "Darf provokant, makaeber und politisch unkorrekt sein — echter Schwarzer Humor. "
                "Themen wie Tod, Krankheit, Gewalt, Katastrophen, Selbstmord, Missgeschicke und Tabuthemen sind erlaubt und gewünscht."
                "Nur der Witz, kein Disclaimer davor oder danach.",
                "Erzaehl mir einen schwarzen Humor Witz."
            )
            await message.reply(f"💀 {result}")
        except Exception as e:
            await message.reply(f"❌ Fehler: {e}")
 
 
 
async def cmd_tldr(message: discord.Message):
    text = message.content[len("!tldr"):].strip()
    # Fallback: wenn kein Text mitgegeben wurde, auf referenzierte Nachricht schauen
    if not text and message.reference and message.reference.resolved:
        text = getattr(message.reference.resolved, "content", "")
    if not text:
        await message.reply("Usage: `!tldr <langer text>` — oder auf eine Nachricht antworten und `!tldr` schreiben.")
        return
    async with message.channel.typing():
        try:
            result = await _one_shot_llm(
                "Fasse den folgenden Text auf Deutsch in 2-3 praegnanten Saetzen zusammen. "
                "Nur die Kernaussagen, nichts hinzudichten.",
                text
            )
            await message.reply(f"📝 **TL;DR:**\n{result}")
        except Exception as e:
            await message.reply(f"❌ Fehler: {e}")
 

async def cmd_roast(message: discord.Message):
    if not _allowed(message.author.id):
        return
    # Ziel-User ermitteln
    if message.mentions:
        target = message.mentions[0]
        name = USER_NAMES.get(target.id, target.display_name)
    else:
        name = USER_NAMES.get(message.author.id, message.author.display_name)
 
    wochentage = ["Montag","Dienstag","Mittwoch","Donnerstag","Freitag","Samstag","Sonntag"]
    now = datetime.now()
    music_ctx = f"{wochentage[now.weekday()]}, {now.strftime('%H:%M')} Uhr"
    if message.guild:
        gm = players.get(message.guild.id)
        if gm and gm.current:
            music_ctx += f" · Spielt: {gm.current.title}"
 
    async with message.channel.typing():
        try:
            reply = await ask_brain(
                message.author.id,
                f"Roaste jetzt {name} auf deine vernichtende, obsessiv-yandere Art. "
                f"Kurz, boese, typisch Basti. Maximal 3 Saetze.",
                music_ctx
            )
            await send_chunked(message, reply)
        except Exception as e:
            await message.reply(f"❌ Fehler: {e}")
  
async def cmd_imagine(message: discord.Message):
    if not _allowed(message.author.id):
        return
    prompt = message.content[len("!imagine"):].strip()
    if not prompt:
        await message.reply("Usage: `!imagine <prompt>`")
        return
    async with message.channel.typing():
        try:
            xc = _get_xai_client()
            response = await xc.image.sample(
                prompt=prompt,
                model=XAI_IMAGE_MODEL,
                aspect_ratio="1:1",
            )
            # Bild herunterladen und direkt in Discord posten
            async with httpx.AsyncClient(timeout=60.0) as http:
                r = await http.get(response.url)
                r.raise_for_status()
                img_bytes = r.content
            file = discord.File(io.BytesIO(img_bytes), filename="basti_imagine.jpg")
            await message.reply(f"🎨 **{prompt[:100]}**", file=file)
        except Exception as e:
            await message.reply(f"❌ Bildgenerierung fehlgeschlagen: {e}")


async def cmd_help(message: discord.Message):
    await message.reply(f"```yml\n{HELP_TEXT_1}\n```")
    await message.reply(f"```yml\n{HELP_TEXT_2}\n```")
 

# ════════════════════════════════════════════════════════════════════════
#  STATUS ROTATION
# ════════════════════════════════════════════════════════════════════════
 
STATUS_ROTATION = [
    (discord.ActivityType.watching,   "!help für alle Befehle"),
    (discord.ActivityType.playing,    "!play <link|ordner>"),
    (discord.ActivityType.listening,  "!randomplay • Zufällige Musik"),
    (discord.ActivityType.playing,    "!explain <thema>"),
    (discord.ActivityType.watching,   "!compare A vs B"),
    (discord.ActivityType.listening,  "!tip <thema>"),
    (discord.ActivityType.playing,    "!tldr <text>"),
    (discord.ActivityType.watching,   "!imagine <prompt> • Bild generieren"),
    (discord.ActivityType.playing,    "!joke • Witz gefällig?"),
    (discord.ActivityType.playing,    "!darkjoke • Schwarzer Humor"),
]
 
 
async def _status_rotation():
    await client.wait_until_ready()
    idx = 0
    while not client.is_closed():
        try:
            playing_title = None
            for gm in players.values():
                if gm.voice and gm.voice.is_playing() and gm.current:
                    playing_title = gm.current.title[:50]
                    break
 
            if playing_title:
                activity = discord.Activity(
                    type=discord.ActivityType.listening,
                    name=playing_title,
                )
            else:
                atype, aname = STATUS_ROTATION[idx % len(STATUS_ROTATION)]
                activity = discord.Activity(type=atype, name=aname)
                idx += 1
 
            await client.change_presence(
                status=discord.Status.online,
                activity=activity,
            )
        except Exception:
            pass   # Netzwerkfehler ignorieren, naechste Runde versuchen
        await asyncio.sleep(20)
 

# ════════════════════════════════════════════════════════════════════════
#  AUTO-DISCONNECT (Leerlauf / leerer Channel)
# ════════════════════════════════════════════════════════════════════════
 
async def _idle_check():
    await client.wait_until_ready()
    while not client.is_closed():
        await asyncio.sleep(60)
        now = time.time()
        for gm in list(players.values()):
            if not (gm.voice and gm.voice.is_connected()):
                continue
            humans = [m for m in gm.voice.channel.members if not m.bot]
            idle = (
                not gm.voice.is_playing()
                and not gm.voice.is_paused()
                and (now - gm.last_activity) > IDLE_TIMEOUT
            )
            if not humans or idle:
                gm.queue.clear()
                gm.intentional_disconnect = True
                try:
                    await gm.voice.disconnect()
                except Exception:
                    pass
                gm.voice = None
                gm.current = None
 
 
# ════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════
 
async def send_chunked(message: discord.Message, text: str):
    if len(text) <= 2000:
        await message.reply(text)
        return
    chunks = [text[i:i + 1990] for i in range(0, len(text), 1990)]
    for i, chunk in enumerate(chunks):
        if i == 0:
            await message.reply(chunk)
        else:
            await message.channel.send(chunk)
 
 
# ════════════════════════════════════════════════════════════════════════
#  EVENTS
# ════════════════════════════════════════════════════════════════════════
 
@client.event
async def on_ready():
    global _idle_task_started, _status_task
    print(f"✅ Basti online als {client.user}  |  Brain: {BRAIN_BACKEND} "
          f"({'an' if brain_enabled else 'aus'})")
    # Status sofort setzen (auch nach Session-Invalidation / Reconnect)
    await client.change_presence(
        status=discord.Status.online,
        activity=discord.Activity(type=discord.ActivityType.watching, name="!help für alle Befehle"),
    )
    if not _idle_task_started:
        client.loop.create_task(_idle_check())
        _idle_task_started = True
    # Status-Task bei jedem on_ready neu starten — wird nach Session-Reset cancelled
    if _status_task is not None:
        _status_task.cancel()
    _status_task = client.loop.create_task(_status_rotation())
 

 
@client.event
async def on_voice_state_update(member: discord.Member,
                                 before: discord.VoiceState,
                                 after: discord.VoiceState):
    """Erkennt erzwungene Voice-Disconnects und versucht automatisch zu reconnecten."""
    if member != client.user:
        return
    # Basti war in einem Channel und ist jetzt draussen
    if before.channel is None or after.channel is not None:
        return
    guild_id = before.channel.guild.id
    gm = players.get(guild_id)
    if gm is None or gm.intentional_disconnect:
        return   # !stop wurde getippt — kein Reconnect
 
    # Kurz warten damit Discord sich beruhigt
    await asyncio.sleep(10)
 
    # Reconnect-Versuch
    try:
        channel = gm.last_channel
        if channel is None:
            return
        gm.voice = await channel.connect()
        gm.intentional_disconnect = False
 
        # Text-Kanal benachrichtigen
        if gm.last_text_channel:
            await gm.last_text_channel.send("🔄 Reconnected nach Verbindungsabbruch.")
 
        # Queue weiter abspielen falls noch Tracks vorhanden
        if gm.queue:
            await start_playing(guild_id)
        else:
            gm.current = None
 
    except Exception as e:
        print(f"⚠️  Reconnect fehlgeschlagen fuer Guild {guild_id}: {e}")
        if gm.last_text_channel:
            try:
                await gm.last_text_channel.send(
                    "❌ Reconnect fehlgeschlagen — bitte manuell `!play` eingeben."
                )
            except Exception:
                pass 
 
@client.event
async def on_message(message: discord.Message):
    global brain_enabled
 
    if message.author.bot:
        return
 
    raw = message.content.strip()
    low = raw.lower()
 
    # ── Musik & Steuerung (offen fuer ALLE, nur auf Servern) ────────────
    if message.guild is not None:
        if low.startswith("!playnext"):
            await cmd_playnext(message); return
        if low.startswith("!play"):
            await cmd_play(message); return
        if low.startswith("!randomplay"):
            await cmd_randomplay(message); return
        if low == "!pause":
            await cmd_pause(message); return
        if low in ("!resume", "!unpause"):
            await cmd_resume(message); return
        if low == "!skip":
            await cmd_skip(message); return
        if low == "!stop":
            await cmd_stop(message); return
        if low == "!clear":
            await cmd_clear(message); return
        if low == "!shuffle":
            await cmd_shuffle(message); return
        if low.startswith("!queue") or low.startswith("!list"):
            await cmd_list(message); return
        if low in ("!np", "!nowplaying"):
            await cmd_np(message); return
        if low.startswith("!volume"):
            await cmd_volume(message); return
        if low.startswith("!remove"):
            await cmd_remove(message); return
        if low.startswith("!move"):
            await cmd_move(message); return
        if low.startswith("!loop"):
            await cmd_loop(message); return
        if low in ("!genres", "!folders"):
            await cmd_genres(message); return
        if low.startswith("!search"):
            await cmd_search(message); return
        if low == "!stats":
            await cmd_stats(message); return
        if low in ("!favourites", "!favorites", "!favs", "!playfav", "!play_favourites"):
            await cmd_favourites(message); return
        if low == "!nightmode":
            await cmd_nightmode(message); return
        if low == "!bassboost":
            await cmd_bassboost(message); return
        if low == "!fade":
            await cmd_fade(message); return
        if low == "!history":
            await cmd_history(message); return
        if low.startswith("!download"):
            await cmd_download(message); return
        if low == "!help":
            await cmd_help(message); return
        if low == "!ping":
            await cmd_ping(message); return
 

    # ── Memory-Commands (nur erlaubte User, auch in DMs) ────────────────
    if low.startswith("!remember"):
        if _allowed(message.author.id):
            fact = message.content[len("!remember"):].strip()
            if fact:
                facts = _load_memory(message.author.id)
                facts.append(fact)
                _save_memory(message.author.id, facts)
                # Chat resetten damit neues Memory sofort gilt
                _xai_chats.pop(message.author.id, None)
                _xai_turns[message.author.id] = 0
                await message.reply(f'🧠 Gespeichert: "{fact}"')
            else:
                await message.reply("Usage: `!remember <fakt>`")
        return
 
    if low == "!memories":
        if _allowed(message.author.id):
            facts = _load_memory(message.author.id)
            if not facts:
                await message.reply("Noch keine Memory-Eintraege.")
            else:
                rows = ["🧠 Deine Memory:", "─" * 40]
                for i, f in enumerate(facts, 1):
                    rows.append(f" {i:>2}. {f}")
                await message.reply("```\n" + "\n".join(rows) + "\n```")
        return
 
    if low.startswith("!forget"):
        if _allowed(message.author.id):
            arg = message.content[len("!forget"):].strip()
            facts = _load_memory(message.author.id)
            try:
                idx = int(arg) - 1
                if not (0 <= idx < len(facts)):
                    raise ValueError
                removed = facts.pop(idx)
                _save_memory(message.author.id, facts)
                _xai_chats.pop(message.author.id, None)
                _xai_turns[message.author.id] = 0
                await message.reply(f'🗑️ Geloescht: "{removed}"')
            except ValueError:
                await message.reply(f"Usage: `!forget <nummer>` — sieh `!memories` fuer Nummern.")
        return

    if low == "!summary":
        await cmd_summary(message); return
    if low in ("!randomfact", "!fact"):
        await cmd_randomfact(message); return
    if low.startswith("!explain"):
        await cmd_explain(message); return
    if low.startswith("!compare"):
        await cmd_compare(message); return
    if low.startswith("!tip"):
        await cmd_tip(message); return
    if low.startswith("!tldr"):
        await cmd_tldr(message); return
    if low.startswith("!roast"):
        await cmd_roast(message); return
    if low.startswith("!imagine"):
        await cmd_imagine(message); return
    if low == "!joke":
        await cmd_joke(message); return
    if low == "!darkjoke":
        await cmd_darkjoke(message); return
 

    # ── Brain-Toggle (nur erlaubte User) ────────────────────────────────
    if low in ("!brain on", "!brain off"):
        if not _allowed(message.author.id):
            return
        brain_enabled = low.endswith("on")
        await message.reply(f"🧠 Brain ist jetzt **{'an' if brain_enabled else 'aus'}**.")
        return
 
    # ── Reden: nur bei Mention/DM, nur erlaubte User ────────────────────
    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = client.user in message.mentions
    if not (is_dm or is_mentioned):
        return
    if not _allowed(message.author.id):
        return
    if not brain_enabled:
        await message.reply("💤 (Brain ist aus – `!brain on` zum Aufwecken.)")
        return
    if not _under_daily_cap():
        await message.reply("🛑 Tageslimit fuer LLM-Calls erreicht.")
        return
 
    content = message.content.replace(f"<@{client.user.id}>", "").strip()
    if not content:
        return

 
    # Live-Musikkontext fuer den Brain-Call zusammenbauen
    wochentage = ["Montag","Dienstag","Mittwoch","Donnerstag","Freitag","Samstag","Sonntag"]
    now = datetime.now()
    music_ctx = f"{wochentage[now.weekday()]}, {now.strftime('%H:%M')} Uhr"
 
    if message.guild:
        gm = players.get(message.guild.id)
        if gm and gm.current:
            src = "lokal" if gm.current.kind == "local" else "YouTube"
            music_ctx += f" · Spielt gerade: {gm.current.title} ({src})"
            if gm.current.kind == "local":
                folder = os.path.basename(os.path.dirname(gm.current.ref))
                music_dir_base = os.path.basename(MUSIC_DIR.rstrip("/\\"))
                if folder.lower() != music_dir_base.lower():
                    music_ctx += f" [Genre: {folder}]"
            if gm.bassboost:
                music_ctx += " · Bassboost"
            if gm.nightmode:
                music_ctx += " · Nightmode"
 
    async with message.channel.typing():
        try:
            reply = await ask_brain(message.author.id, content, music_ctx)
            _save_conversation(message.author.id, content, reply)
            await send_chunked(message, reply)
        except httpx.TimeoutException:
            await message.reply("⏱️ Brain antwortet nicht...")
        except Exception as e:
            await message.reply(f"❌ Fehler: {e}")
 
 
 
if __name__ == "__main__":
    client.run(DISCORD_TOKEN)