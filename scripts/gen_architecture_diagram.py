"""
Generate system architecture diagram using PIL/Pillow.
Output: docs/architecture.png
"""
from PIL import Image, ImageDraw, ImageFont
import os

# ─── Canvas ───────────────────────────────────────────────────────────────────
W, H = 1920, 1480
BG   = (10, 14, 22)
img  = Image.new("RGB", (W, H), BG)
draw = ImageDraw.Draw(img)

# ─── Fonts ────────────────────────────────────────────────────────────────────
FONT_BASE = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

def fnt(size, bold=False):
    try:
        return ImageFont.truetype(FONT_BOLD if bold else FONT_BASE, size)
    except Exception:
        return ImageFont.load_default()

F_TITLE = fnt(30, bold=True)
F_SECT  = fnt(17, bold=True)
F_BODY  = fnt(14)
F_SMALL = fnt(12)
F_TINY  = fnt(11)

# ─── Palette ──────────────────────────────────────────────────────────────────
C_INPUT  = ( 34, 197,  94)   # green    — sensors / input
C_DETECT = ( 16, 185, 129)   # emerald  — detection / SR
C_DB     = ( 99, 102, 241)   # indigo   — databases
C_PAIR   = (  6, 182, 212)   # cyan     — pairing
C_LLM    = (168,  85, 247)   # purple   — LLM / reporting
C_API    = ( 20, 184, 166)   # teal     — web API
C_UI     = ( 59, 130, 246)   # blue     — dashboard
C_CFG    = (245, 158,  11)   # amber    — config / RAG
C_ARROW  = (148, 163, 184)   # slate-400
C_TXT    = (226, 232, 240)   # slate-200
C_DIM    = (100, 116, 139)   # slate-500
C_LINE   = ( 30,  41,  59)   # slate-800

# ─── Drawing helpers ──────────────────────────────────────────────────────────
def box(xy, fill, outline, r=10, bw=2):
    draw.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=bw)

def ctext(cx, cy, text, font, color=C_TXT):
    bb = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    draw.text((cx - tw//2, cy - th//2), text, font=font, fill=color)

def arrow_down(x, y0, y1, color=C_ARROW):
    """Vertical arrow pointing downward from y0 to y1."""
    draw.line([(x, y0), (x, y1)], fill=color, width=2)
    # arrowhead pointing down
    draw.polygon([(x-7, y1-12), (x+7, y1-12), (x, y1)], fill=color)

def arrow_left(x0, x1, y, color=C_ARROW):
    """Horizontal arrow pointing LEFT: from x0 → x1 (x1 < x0)."""
    draw.line([(x0, y), (x1, y)], fill=color, width=2)
    # arrowhead pointing left: tip at (x1, y), base at (x1+12, y±7)
    draw.polygon([(x1+12, y-7), (x1+12, y+7), (x1, y)], fill=color)

def hline(y, color=C_LINE):
    draw.line([(60, y), (W-60, y)], fill=color, width=1)

def vline(x, y0, y1, color=C_LINE):
    draw.line([(x, y0), (x, y1)], fill=color, width=1)

# ─── Main pipeline dimensions ─────────────────────────────────────────────────
CX   = 430          # main pipeline centre x
BW   = 360          # box width  (x: 250 → 610)
BH   = 60           # standard box half-height  (box y span: cy-BH//2 → cy+BH//2)
LX   = CX - BW//2  # left  edge of main boxes = 250
RX   = CX + BW//2  # right edge of main boxes = 610

# Right panels
CXR  = 1380         # right column centre x
BWR  = 480          # right panel width  (x: 1140 → 1620)
LXR  = CXR - BWR//2 # left  edge of right panels = 1140
RXR  = CXR + BWR//2 # right edge               = 1620

# Arrow corridor: x ∈ [RX+4, LXR-4]  →  [614, 1136]
# Horizontal arrows go from LXR to RX (pointing left)

DIVIDER_X = 870

# ─── Row Y centres for main pipeline ─────────────────────────────────────────
Y = {
    "title"   :  38,
    "input"   : 130,  # 4 input boxes
    "detect"  : 268,  # SAM3 Detection
    "sensdb"  : 398,  # Sensor DB
    "pair"    : 540,  # Temporal Pairing (taller)
    "pairdb"  : 668,  # Pairing DB
    "report"  : 798,  # Reporting Layer (taller)
    "repdb"   : 928,  # Reports DB
    "webapi"  :1050,  # Web API
    "dash"    :1180,  # Dashboard
}

# ─── Right panel Y centres ────────────────────────────────────────────────────
YR = {
    "sr"      : 268,  # Super-Resolution  (aligned with detect)
    "config"  : 480,  # Config            (spans detect→pair)
    "clip"    : 700,  # CLIP Tracker      (aligned with pair/pairdb)
    "rag"     : 900,  # Doctrine RAG      (aligned with report)
}

# ─── Helper: draw a main-pipeline box ─────────────────────────────────────────
def main_box(cy, half_h, fill_rgb, outline, title, subtitles=()):
    y0, y1 = cy - half_h, cy + half_h
    box([LX, y0, RX, y1], fill=fill_rgb, outline=outline)
    if subtitles:
        ctext(CX, y0 + 22, title, F_SECT, outline)
        for i, s in enumerate(subtitles):
            ctext(CX, y0 + 44 + i*20, s, F_SMALL, C_DIM)
    else:
        ctext(CX, cy, title, F_SECT, outline)
    return y0, y1

# ─── Helper: draw a right-side panel ──────────────────────────────────────────
def right_panel(cy, height, outline, fill_rgb, title, lines=()):
    y0, y1 = cy - height//2, cy + height//2
    box([LXR, y0, RXR, y1], fill=fill_rgb, outline=outline)
    ctext(CXR, y0 + 20, title, F_SECT, outline)
    for i, l in enumerate(lines):
        ctext(CXR, y0 + 46 + i*20, l, F_SMALL, C_DIM)
    return y0, y1, (y0+y1)//2

# ═══════════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════════
ctext(W//2, Y["title"], "Multi-Source Intelligent System (MSIS)  —  Architecture", F_TITLE, C_TXT)
hline(64)

# ─── Column headers ───────────────────────────────────────────────────────────
ctext(CX,  76, "Main Pipeline", F_SMALL, C_DIM)
ctext(CXR, 76, "Supporting Components", F_SMALL, C_DIM)
vline(DIVIDER_X, 64, H-80)

# ═══════════════════════════════════════════════════════════════════════════════
# INPUT SOURCES  (4 boxes, centred on CX)
# ═══════════════════════════════════════════════════════════════════════════════
input_items = [
    ("Satellite\nImagery", C_INPUT),
    ("Drone\nImagery",     C_INPUT),
    ("GPS /\nMetadata",    C_INPUT),
    ("Manual\nUpload",     C_INPUT),
]
ibw, igap = 148, 16   # box width, gap between boxes
total_in  = len(input_items) * ibw + (len(input_items)-1) * igap   # =636
ix_start  = CX - total_in//2                                        # =112

in_tops = []
for k, (label, col) in enumerate(input_items):
    ix0 = ix_start + k * (ibw + igap)
    ix1 = ix0 + ibw
    iy0, iy1 = Y["input"]-28, Y["input"]+28
    in_tops.append((ix0, ix1))
    box([ix0, iy0, ix1, iy1], fill=(20, 28, 44), outline=col, bw=2)
    lines = label.split("\n")
    cy_box = (iy0+iy1)//2
    if len(lines) == 2:
        ctext((ix0+ix1)//2, cy_box - 9, lines[0], F_BODY, col)
        ctext((ix0+ix1)//2, cy_box + 9, lines[1], F_BODY, col)
    else:
        ctext((ix0+ix1)//2, cy_box, label, F_BODY, col)

# Funnel: vertical lines from each input box down to collecting horizontal bar,
# then single arrow down to Detection box.
COLLECT_Y = Y["input"] + 44
for ix0, ix1 in in_tops:
    icx = (ix0+ix1)//2
    draw.line([(icx, Y["input"]+28), (icx, COLLECT_Y)], fill=C_ARROW, width=2)

# Horizontal collecting bar
bar_x0 = (in_tops[0][0] + in_tops[0][1])//2
bar_x1 = (in_tops[-1][0] + in_tops[-1][1])//2
draw.line([(bar_x0, COLLECT_Y), (bar_x1, COLLECT_Y)], fill=C_ARROW, width=2)

# Single arrow from centre of collecting bar down to Detection box
arrow_down(CX, COLLECT_Y, Y["detect"] - 32)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE BOXES
# ═══════════════════════════════════════════════════════════════════════════════

# 1. SAM3 Detection
_, det_y1 = main_box(Y["detect"], 30,
    fill_rgb=(12, 28, 20), outline=C_DETECT,
    title="SAM3  (facebook/sam3)",
    subtitles=("Text-prompted concept segmentation",))
arrow_down(CX, det_y1, Y["sensdb"] - 32)

# 2. Sensor DB
_, sdb_y1 = main_box(Y["sensdb"], 30,
    fill_rgb=(16, 18, 48), outline=C_DB,
    title="Sensor DB  (SQLite)",
    subtitles=("image_records · detection_records",))
arrow_down(CX, sdb_y1, Y["pair"] - 42)

# 3. Temporal Pairing
_, pair_y1 = main_box(Y["pair"], 40,
    fill_rgb=(6, 28, 36), outline=C_PAIR,
    title="Temporal Pairing Engine",
    subtitles=("Mode A: SAM3 video predictor tracker",
               "Mode B: CLIP cosine + geo proximity"))
arrow_down(CX, pair_y1, Y["pairdb"] - 32)

# 4. Pairing DB
_, pdb_y1 = main_box(Y["pairdb"], 30,
    fill_rgb=(16, 18, 48), outline=C_DB,
    title="Pairing DB  (SQLite)",
    subtitles=("new · matched · moved · disappeared",))
arrow_down(CX, pdb_y1, Y["report"] - 40)

# 5. Reporting
_, rep_y1 = main_box(Y["report"], 40,
    fill_rgb=(24, 12, 44), outline=C_LLM,
    title="Reporting Layer",
    subtitles=("EXAONE4-32b  (vLLM / Ollama)",
               "Intelligence report + KO translation"))
arrow_down(CX, rep_y1, Y["repdb"] - 32)

# 6. Reports DB
_, rdb_y1 = main_box(Y["repdb"], 30,
    fill_rgb=(16, 18, 48), outline=C_DB,
    title="Reports DB  (SQLite)",
    subtitles=("report_records · analysis sessions",))
arrow_down(CX, rdb_y1, Y["webapi"] - 32)

# 7. Web API
_, wapi_y1 = main_box(Y["webapi"], 30,
    fill_rgb=(6, 26, 28), outline=C_API,
    title="Web API  (Flask + aiohttp)",
    subtitles=("REST /api/*  ·  /stream  ·  /ws",))
arrow_down(CX, wapi_y1, Y["dash"] - 40)

# 8. Dashboard
main_box(Y["dash"], 40,
    fill_rgb=(6, 16, 40), outline=C_UI,
    title="Dashboard  (HTML / JS)",
    subtitles=("Leaflet map  ·  Report viewer",
               "Matching visualizer  ·  Mobile-responsive"))

# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT PANELS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Super-Resolution ──────────────────────────────────────────────────────────
sr_y0, sr_y1, sr_cy = right_panel(
    YR["sr"], 100, C_DETECT, (10, 24, 16),
    "Super-Resolution  (pre-processing)",
    ["Real-ESRGAN  →  PIL LANCZOS fallback",
     "Target: 8 000 × 6 000 px before SAM3"])
# Arrow: SR → Detection (horizontal left)
arrow_left(LXR, RX + 2, sr_cy, C_DETECT)

# ── Config ────────────────────────────────────────────────────────────────────
cfg_y0, cfg_y1, cfg_cy = right_panel(
    YR["config"], 280, C_CFG, (22, 16, 6),
    "Config  (src/config.py)",
    ["SAM3_MODEL_NAME  ·  SAM3_DEVICE",
     "TILE_ENABLED · TILE_SIZE · TILE_OVERLAP",
     "MAX_BBOX_AREA_RATIO  (0.15 / 0.70)",
     "TRACKING_MODE  (sam3_tracker | similarity)",
     "LLM_BACKEND  (vllm | ollama)",
     "COORD_MATCH_RADIUS · MOVE_THRESHOLD",
     "DOCTRINE_ENABLED  ·  DOCTRINE_TOP_K",
     "LLM_TENSOR_PARALLEL_SIZE"])
# Arrow: Config → Sensor DB level (horizontal left)
arrow_left(LXR, RX + 2, cfg_cy, C_CFG)

# ── CLIP / Similarity Tracker ─────────────────────────────────────────────────
clip_y0, clip_y1, clip_cy = right_panel(
    YR["clip"], 160, C_PAIR, (6, 24, 32),
    "Similarity Tracker  (Mode B)",
    ["CLIP  openai/clip-vit-base-patch16",
     "Mask-crop → image embedding",
     "cosine sim · geo proximity · size sim",
     "Static class geo pre-matching (Step 0)"])
# Arrow: CLIP panel → Pairing box (horizontal left)
arrow_left(LXR, RX + 2, clip_cy, C_PAIR)

# ── Doctrine RAG ──────────────────────────────────────────────────────────────
rag_y0, rag_y1, rag_cy = right_panel(
    YR["rag"], 160, C_CFG, (22, 16, 6),
    "Doctrine RAG  (FAISS)",
    ["Military doctrine documents",
     "FAISS vector index  ·  SBERT embeddings",
     "Top-K chunks → LLM prompt context",
     "build_doctrine_vectordb.py"])
# Arrow: Doctrine RAG → Reporting (horizontal left)
arrow_left(LXR, RX + 2, rag_cy, C_CFG)

# ═══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════════════════════
hline(H - 80)
legend = [
    ("Input / Sensors",   C_INPUT),
    ("Detection (SAM3)",  C_DETECT),
    ("Database (SQLite)", C_DB),
    ("Pairing Engine",    C_PAIR),
    ("LLM / Reporting",   C_LLM),
    ("Web API",           C_API),
    ("Dashboard / UI",    C_UI),
    ("Config / RAG",      C_CFG),
]
ly = H - 56
draw.text((60, ly - 18), "Legend:", font=F_SMALL, fill=C_DIM)
spc = (W - 140) // len(legend)
for i, (lbl, col) in enumerate(legend):
    lx = 60 + i * spc
    draw.rounded_rectangle([lx, ly, lx+13, ly+13], radius=3, fill=col)
    draw.text((lx + 18, ly), lbl, font=F_TINY, fill=C_TXT)

draw.text((W - 620, H - 28),
          "github.com/Parkdev22222/multi-source-intelligent-system",
          font=F_TINY, fill=C_DIM)

# ─── Save ─────────────────────────────────────────────────────────────────────
out_dir  = os.path.join(os.path.dirname(__file__), "..", "docs")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "architecture.png")
img.save(out_path, "PNG", optimize=True)
print(f"Saved: {out_path}  ({W}×{H})")
