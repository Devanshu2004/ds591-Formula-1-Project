import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging
import fsspec
from typing import Optional, Dict, List, Tuple

# ── Global Constants ───────────────────────────────────────────

RACE_LOCATIONS = [
    "Italian Grand Prix", "Azerbaijan Grand Prix", "Singapore Grand Prix",
    "Mexico City Grand Prix", "Brazilian Grand Prix", "Las Vegas Grand Prix",
    "Qatar Grand Prix", "Abu Dhabi Grand Prix", "United States Grand Prix",
    "Australian Grand Prix", "Austrian Grand Prix", "Bahrain Grand Prix",
    "Belgian Grand Prix", "British Grand Prix", "Canadian Grand Prix",
    "Chinese Grand Prix", "Dutch Grand Prix", "Eifel Grand Prix",
    "Emilia Romagna Grand Prix", "French Grand Prix", "German Grand Prix",
    "Hungarian Grand Prix", "Japanese Grand Prix", "Miami Grand Prix",
    "Monaco Grand Prix", "Portuguese Grand Prix", "Russian Grand Prix",
    "Sakhir Grand Prix", "Saudi Arabian Grand Prix", "Spanish Grand Prix",
    "Styrian Grand Prix", "São Paulo Grand Prix", "Turkish Grand Prix",
    "Tuscan Grand Prix"
]

DRIVER_ABB = {
    "ALB": ["Alexander Albon", "Alex Albon"],
    "ALO": ["Fernando Alonso"],
    "ANT": ["Kimi Antonelli"],
    "BEA": ["Oliver Bearman"],
    "BOR": ["Gabriel Bortoleto"],
    "BOT": ["Valtteri Bottas"],
    "COL": ["Franco Colapinto"],
    "DEV": ["Nyck De Vries"],
    "DOO": ["Jack Doohan"],
    "ERI": ["Marcus Ericsson"],
    "FIT": ["Pietro Fittipaldi"],
    "GAS": ["Pierre Gasly"],
    "GIO": ["Antonio Giovinazzi"],
    "GRO": ["Romain Grosjean"],
    "HAD": ["Isack Hadjar"],
    "HAM": ["Lewis Hamilton"],
    "HAR": ["Brendon Hartley"],
    "HUL": ["Nico Hülkenberg"],
    "KUB": ["Robert Kubica"],
    "KVY": ["Daniil Kvyat"],
    "LAT": ["Nicholas Latifi"],
    "LAW": ["Liam Lawson"],
    "LEC": ["Charles Leclerc"],
    "MAG": ["Kevin Magnussen"],
    "MSC": ["Mick Schumacher"],
    "MAZ": ["Nikita Mazepin"],
    "NOR": ["Lando Norris"],
    "OCO": ["Esteban Ocon"],
    "PER": ["Sergio Perez"],
    "PIA": ["Oscar Piastri"],
    "RAI": ["Kimi Räikkönen"],
    "RIC": ["Daniel Ricciardo"],
    "RUS": ["George Russell"],
    "SAI": ["Carlos Sainz", "Carlos Sainz Jr."],
    "SAR": ["Logan Sargeant"],
    "SIR": ["Sergey Sirotkin"],
    "STR": ["Lance Stroll"],
    "TSU": ["Yuki Tsunoda"],
    "VAN": ["Stoffel Vandoorne"],
    "VER": ["Max Verstappen"],
    "VET": ["Sebastian Vettel"],
    "ZHO": ["Zhou Guanyu"]
}

DRIVER_LIST = list(DRIVER_ABB.keys())


# ── Configuration ──────────────────────────────────────────────

SAMPLING_RATE      = 0.1
WINDOW_SECONDS     = 30
HORIZON_SECONDS    = 300

SEQUENCE_LENGTH    = int(WINDOW_SECONDS  / SAMPLING_RATE)   # 300
PREDICTION_HORIZON = int(HORIZON_SECONDS / SAMPLING_RATE)   # 3000

LSTM_HIDDEN_SIZE   = 64
LSTM_LAYERS        = 2
EMBEDDING_DIM      = 32
DROPOUT            = 0.3
BATCH_SIZE         = 32
EPOCHS             = 50
LEARNING_RATE      = 1e-3
NUM_POSITIONS      = 20

SESSION_TYPE       = os.getenv("SESSION_TYPE", "R")


# ── Azure Storage Configuration ────────────────────────────────

STORAGE_ACCOUNT_NAME = os.getenv("STORAGE_ACCOUNT_NAME", "formula1analyticsdata")
STORAGE_ACCOUNT_KEY  = os.getenv("STORAGE_ACCOUNT_KEY")
GOLD_CONTAINER       = os.getenv("GOLD_CONTAINER",  "gold")
MODEL_CONTAINER      = os.getenv("MODEL_CONTAINER", "platinum")


def get_storage_options() -> Dict:
    return {
        "account_name": STORAGE_ACCOUNT_NAME,
        "account_key":  STORAGE_ACCOUNT_KEY,
    }


def get_fs() -> fsspec.AbstractFileSystem:
    return fsspec.filesystem(
        "abfs",
        account_name=STORAGE_ACCOUNT_NAME,
        account_key=STORAGE_ACCOUNT_KEY,
    )


def abfs_path(container: str, path: str) -> str:
    return (
        f"abfs://{container}"
        f"@{STORAGE_ACCOUNT_NAME}.dfs.core.windows.net/{path}"
    )


def blob_path(container: str, path: str) -> str:
    return f"{container}/{path}"


# ── Column Definitions ─────────────────────────────────────────

TARGET_COL        = "target_pos"
SOCIAL_SCORE_COL  = "social_life_score"       # updated name
SESSION_TIME_COL  = "session_time"
RADIO_FLAG_COL    = "radio_data_available"    # 0 = no radio, 1 = radio active
RACE_GROUP_COLS   = ["race_id", "race_year", "race_location"]

# Columns to drop — string name cols and metadata
DROP_COLS = {
    "target_driver",        # string name — use target_driver_number instead
    "driver_ahead",         # string name — use driver_ahead_number instead
    "driver_behind",        # string name — use driver_behind_number instead
    "target_rpm",           # dropped per instruction
    "driver_ahead_rpm",     # dropped per instruction
    "driver_behind_rpm",    # dropped per instruction
    "race_date",            # metadata
}

# Meta cols — used for grouping/sorting/target/social but NOT fed to LSTM
META_COLS = {
    SESSION_TIME_COL,
    "race_id",
    "race_date",
    "race_year",
    "race_location",
    TARGET_COL,
    SOCIAL_SCORE_COL,
    RADIO_FLAG_COL,         # used as attention gate, not as LSTM feature
}

# Radio feature columns — all prefixed with radio_ (excluding the flag)
# These are NaN when radio_data_available = 0
# We use radio_data_available as the sparse attention gate
RADIO_COLS = [
    # Quality / confidence
    "radio_transcript_quality",
    "radio_confidence",
    "radio_action_required",

    # Primary event type
    "radio_primary_event_type_celebration",
    "radio_primary_event_type_damage_issue",
    "radio_primary_event_type_defending",
    "radio_primary_event_type_information_only",
    "radio_primary_event_type_mechanical_issue",
    "radio_primary_event_type_overtaking",
    "radio_primary_event_type_pace_management",
    "radio_primary_event_type_pit_call",
    "radio_primary_event_type_safety",
    "radio_primary_event_type_tire_strategy",
    "radio_primary_event_type_traffic",
    "radio_primary_event_type_weather",

    # Action type
    "radio_action_type_acknowledge_info",
    "radio_action_type_conserve",
    "radio_action_type_defend",
    "radio_action_type_manage_tires",
    "radio_action_type_overtake",
    "radio_action_type_pit_now",
    "radio_action_type_pit_soon",
    "radio_action_type_push",
    "radio_action_type_report_issue",
    "radio_action_type_stay_out",
    "radio_action_type_unknown",

    # Secondary event type
    "radio_secondary_event_type_celebration",
    "radio_secondary_event_type_damage_issue",
    "radio_secondary_event_type_defending",
    "radio_secondary_event_type_information_only",
    "radio_secondary_event_type_mechanical_issue",
    "radio_secondary_event_type_overtaking",
    "radio_secondary_event_type_pace_management",
    "radio_secondary_event_type_pit_call",
    "radio_secondary_event_type_safety",
    "radio_secondary_event_type_tire_strategy",
    "radio_secondary_event_type_traffic",
    "radio_secondary_event_type_weather",

    # Car issue signals
    "radio_car_issue_signal.has_issue",
    "radio_car_issue_signal.issue_type_battery",
    "radio_car_issue_signal.issue_type_brakes",
    "radio_car_issue_signal.issue_type_engine",
    "radio_car_issue_signal.issue_type_floor_damage",
    "radio_car_issue_signal.issue_type_gearbox",
    "radio_car_issue_signal.issue_type_none",
    "radio_car_issue_signal.issue_type_overheating",
    "radio_car_issue_signal.issue_type_steering",
    "radio_car_issue_signal.issue_type_unknown",
    "radio_car_issue_signal.issue_type_wing_damage",
    "radio_car_issue_signal.severity_minor",
    "radio_car_issue_signal.severity_moderate",
    "radio_car_issue_signal.severity_none",

    # Strategy signals
    "radio_strategy_signal.pit_related",
    "radio_strategy_signal.tire_related",
    "radio_strategy_signal.fuel_saving",
    "radio_strategy_signal.pace_change",
    "radio_strategy_signal.weather_related",
    "radio_strategy_signal.safety_related",

    # Racecraft signals
    "radio_racecraft_signal.traffic_mentioned",
    "radio_racecraft_signal.overtake_mentioned",
    "radio_racecraft_signal.defend_mentioned",
    "radio_racecraft_signal.drs_mentioned",
    "radio_racecraft_signal.gap_management_mentioned",
]


# ── Azure Data Loader ──────────────────────────────────────────

def load_driver_parquet(driver: str) -> Optional[pd.DataFrame]:
    """
    Load driver gold parquet from Azure.
    Path: gold/{SESSION_TYPE}/{driver}_gold.parquet
    """
    path = abfs_path(
        GOLD_CONTAINER,
        f"{SESSION_TYPE}/{driver}_gold.parquet"
    )
    try:
        df = pd.read_parquet(path, storage_options=get_storage_options())
        logging.info(f"Loaded {driver}: {df.shape[0]} rows")
        return df
    except Exception as e:
        logging.warning(f"Could not load {driver}: {e}")
        return None


def save_model_to_azure(local_path: str, blob_name: str):
    """Upload local model file to platinum/models/"""
    try:
        fs     = get_fs()
        remote = blob_path(MODEL_CONTAINER, blob_name)
        fs.put(local_path, remote)
        logging.info(f"Uploaded: {MODEL_CONTAINER}/{blob_name}")
    except Exception as e:
        logging.warning(f"Could not upload {blob_name}: {e}")


def load_model_from_azure(blob_name: str, local_path: str):
    """Download model file from platinum/models/ to local disk."""
    try:
        fs     = get_fs()
        remote = blob_path(MODEL_CONTAINER, blob_name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        fs.get(remote, local_path)
        logging.info(f"Downloaded: {MODEL_CONTAINER}/{blob_name}")
    except Exception as e:
        logging.warning(f"Could not download {blob_name}: {e}")


# ── Dataset ────────────────────────────────────────────────────

class DriverDataset(Dataset):
    """
    Builds sequences from a driver's gold DataFrame.
    All features assumed numerical and ready — no preprocessing.

    Key design decisions:
    - radio_data_available is used as the sparse attention gate signal,
      NOT as a feature fed into the Bi-LSTM
    - All radio feature cols are NaN when radio_data_available = 0,
      filled with 0.0 before feeding to the attention module
    - social_life_score is NaN where no social data exists,
      filled with 5.0 (neutral midpoint of 1-10 scale)
    - DROP_COLS and string name cols are excluded before building sequences
    - Dynamic OHE columns (team, compound, gear, track_status) are
      auto-detected from CSV headers — no hardcoding needed

    Sliding window:
        - Group by (race_id, race_year, race_location)
        - Sort by session_time within each race
        - Window = SEQUENCE_LENGTH = 300 timesteps (30 seconds)
        - Target = position at timestep
                   i + SEQUENCE_LENGTH + PREDICTION_HORIZON - 1
                   (5 minutes = 3000 timesteps ahead)
    """

    def __init__(self, df: pd.DataFrame):
        # Drop unwanted columns
        drop     = DROP_COLS | META_COLS
        df_clean = df.drop(columns=[c for c in drop if c in df.columns])

        # Identify radio columns present in this DataFrame
        present_radio = [c for c in RADIO_COLS if c in df_clean.columns]
        radio_set     = set(present_radio)

        # Feature columns: everything except radio cols
        # Dynamic OHE cols (team/compound/gear/track_status) are
        # automatically included here
        self.feature_cols = sorted([
            c for c in df_clean.columns if c not in radio_set
        ])
        self.radio_cols = present_radio

        self.sequences       = []
        self.targets         = []
        self.social_scores   = []
        self.radio_sequences = []
        self.radio_gates     = []   # 1.0 where radio active, 0.0 where silent

        group_keys = [c for c in RACE_GROUP_COLS if c in df.columns]

        for _, race_df in df.groupby(group_keys):
            race_df = race_df.sort_values(SESSION_TIME_COL).reset_index(drop=True)

            features  = race_df[self.feature_cols].fillna(0).values.astype(np.float32)
            positions = race_df[TARGET_COL].values.astype(np.float32)

            # Social score: NaN → 5.0 (neutral midpoint)
            social = float(
                race_df[SOCIAL_SCORE_COL].fillna(5.0).iloc[0]
                if SOCIAL_SCORE_COL in race_df.columns
                else 5.0
            )

            # Radio gate: 1.0 where radio_data_available = 1, else 0.0
            if RADIO_FLAG_COL in race_df.columns:
                gate = race_df[RADIO_FLAG_COL].fillna(0).values.astype(np.float32)
            else:
                gate = np.zeros(len(race_df), dtype=np.float32)

            # Radio features: NaN → 0.0 (no event)
            if present_radio:
                radio = race_df[present_radio].fillna(0).values.astype(np.float32)
            else:
                radio = np.zeros((len(race_df), 1), dtype=np.float32)

            n            = len(features)
            min_required = SEQUENCE_LENGTH + PREDICTION_HORIZON

            if n < min_required:
                logging.warning(
                    f"Race {race_df[group_keys].iloc[0].to_dict()} "
                    f"has {n} timesteps (need {min_required}) — skipping"
                )
                continue

            for i in range(n - SEQUENCE_LENGTH - PREDICTION_HORIZON + 1):
                target_idx = i + SEQUENCE_LENGTH + PREDICTION_HORIZON - 1
                target     = positions[target_idx]

                if np.isnan(target) or target < 1 or target > NUM_POSITIONS:
                    continue

                self.sequences.append(features[i: i + SEQUENCE_LENGTH])
                self.targets.append(int(target) - 1)   # 0-indexed
                self.social_scores.append(social)
                self.radio_sequences.append(radio[i: i + SEQUENCE_LENGTH])
                self.radio_gates.append(gate[i: i + SEQUENCE_LENGTH])

        logging.info(
            f"Dataset: {len(self.sequences)} sequences | "
            f"input_size={len(self.feature_cols)} | "
            f"radio_dim={len(self.radio_cols) or 1}"
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx]),
            torch.tensor(self.targets[idx],       dtype=torch.long),
            torch.tensor(self.social_scores[idx], dtype=torch.float32),
            torch.tensor(self.radio_sequences[idx]),
            torch.tensor(self.radio_gates[idx]),  # (seq_len,) gate signal
        )


# ── Radio Attention ────────────────────────────────────────────

class RadioAttention(nn.Module):
    """
    Sparse attention gate for radio events.

    Uses radio_data_available as the gate signal — cleaner and more
    reliable than checking if radio feature values are zero.

    gate = 0.0 → silent timestep → hidden state unchanged
    gate = 1.0 → radio event    → hidden state modulated by radio content
    """

    def __init__(self, hidden_size: int, radio_dim: int):
        super().__init__()
        self.radio_proj = nn.Linear(radio_dim, hidden_size * 2)
        self.gate_proj  = nn.Linear(hidden_size * 2, 1)
        self.sigmoid    = nn.Sigmoid()

    def forward(
        self,
        lstm_out:   torch.Tensor,  # (batch, seq_len, hidden*2)
        radio:      torch.Tensor,  # (batch, seq_len, radio_dim)
        radio_gate: torch.Tensor,  # (batch, seq_len) — from radio_data_available
    ) -> torch.Tensor:
        # Expand gate to (batch, seq_len, 1)
        gate_mask = radio_gate.unsqueeze(-1)

        # Project radio features to hidden space
        proj = torch.tanh(self.radio_proj(radio))

        # Compute soft attention weight
        attn = self.sigmoid(self.gate_proj(lstm_out + proj))

        # Apply only at active radio timesteps
        return lstm_out + (attn * proj * gate_mask)


# ── Driver Channel (Bi-LSTM) ───────────────────────────────────

class DriverChannel(nn.Module):
    """
    Bi-LSTM channel for a single driver.

    input_size : auto-detected from gold DataFrame feature columns
    radio_dim  : auto-detected from gold DataFrame radio columns

    Forward returns:
        embedding : (batch, EMBEDDING_DIM)  →  Random Forest input
        logits    : (batch, NUM_POSITIONS)  →  channel training loss
    """

    def __init__(self, input_size: int, radio_dim: int):
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if LSTM_LAYERS > 1 else 0.0,
        )

        self.radio_attention = RadioAttention(LSTM_HIDDEN_SIZE, radio_dim)

        # Social score injected as additive bias on final hidden state
        self.social_proj = nn.Linear(1, LSTM_HIDDEN_SIZE * 2)

        self.dropout = nn.Dropout(DROPOUT)

        self.embedding_proj = nn.Sequential(
            nn.Linear(LSTM_HIDDEN_SIZE * 2, EMBEDDING_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )

        self.classifier = nn.Linear(EMBEDDING_DIM, NUM_POSITIONS)

    def forward(
        self,
        x:          torch.Tensor,  # (batch, seq_len, input_size)
        social:     torch.Tensor,  # (batch,)
        radio:      torch.Tensor,  # (batch, seq_len, radio_dim)
        radio_gate: torch.Tensor,  # (batch, seq_len)
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        lstm_out, _ = self.bilstm(x)
        # (batch, seq_len, hidden*2)

        lstm_out = self.radio_attention(lstm_out, radio, radio_gate)

        last = lstm_out[:, -1, :]
        # (batch, hidden*2)

        # Add social score as bias (NaN already filled with 5.0 in Dataset)
        last = last + torch.tanh(self.social_proj(social.unsqueeze(-1)))

        last   = self.dropout(last)
        emb    = self.embedding_proj(last)   # (batch, EMBEDDING_DIM)
        logits = self.classifier(emb)        # (batch, NUM_POSITIONS)

        return emb, logits


# ── Train One Driver Channel ───────────────────────────────────

def train_driver_channel(
    driver:   str,
    save_dir: str,
    device:   torch.device,
) -> bool:
    """
    Load driver gold parquet from Azure,
    train Bi-LSTM channel, save weights locally and
    upload to platinum/models/channel_{driver}.pt
    """
    logging.info(f"Training channel: {driver}")

    df = load_driver_parquet(driver)
    if df is None or df.empty:
        logging.warning(f"No data for {driver} — skipping")
        return False

    dataset = DriverDataset(df)
    if len(dataset) == 0:
        logging.warning(f"No sequences for {driver} — skipping")
        return False

    input_size = len(dataset.feature_cols)
    radio_dim  = len(dataset.radio_cols) if dataset.radio_cols else 1

    logging.info(
        f"  {driver}: {len(dataset)} sequences | "
        f"input_size={input_size} | radio_dim={radio_dim}"
    )

    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model     = DriverChannel(input_size, radio_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5
    )
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for x_b, y_b, s_b, r_b, g_b in loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            s_b = s_b.to(device)
            r_b = r_b.to(device)
            g_b = g_b.to(device)

            optimizer.zero_grad()
            _, logits = model(x_b, s_b, r_b, g_b)
            loss      = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            logging.info(
                f"  [{driver}] Epoch {epoch + 1}/{EPOCHS}  "
                f"Loss: {total_loss / len(loader):.4f}"
            )

    # Save locally
    os.makedirs(save_dir, exist_ok=True)
    local_path = os.path.join(save_dir, f"channel_{driver}.pt")
    torch.save(
        {
            "state_dict":   model.state_dict(),
            "input_size":   input_size,
            "radio_dim":    radio_dim,
            "feature_cols": dataset.feature_cols,
            "radio_cols":   dataset.radio_cols,
        },
        local_path,
    )
    logging.info(f"Saved locally: {local_path}")

    # Upload to platinum/models/channel_{driver}.pt
    save_model_to_azure(local_path, f"models/channel_{driver}.pt")

    return True


# ── Train All Driver Channels ──────────────────────────────────

def train_all_channels(
    save_dir: str,
    device:   torch.device,
):
    """
    Train all driver channels using DRIVER_LIST from DRIVER_ABB.
    Loads each driver's gold parquet from Azure gold container.
    Saves each channel to platinum/models/channel_{driver}.pt
    """
    trained, skipped = [], []

    for driver in DRIVER_LIST:
        success = train_driver_channel(driver, save_dir, device)
        if success:
            trained.append(driver)
        else:
            skipped.append(driver)

    logging.info(f"Trained : {trained}")
    logging.info(f"Skipped : {skipped}")


# ── Extract Embeddings for RF ──────────────────────────────────

def extract_embeddings(
    save_dir: str,
    device:   torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load each saved channel, extract embeddings over full dataset,
    align to minimum sample count, concatenate into RF input matrix.

    Row i of X = [emb_driver1[i], ..., emb_driverN[i]]
    y[i]        = target position (0-indexed) from first available driver
    """
    all_embeddings: Dict[str, np.ndarray] = {}
    all_targets:    Dict[str, np.ndarray] = {}

    for driver in DRIVER_LIST:
        local_path = os.path.join(save_dir, f"channel_{driver}.pt")

        if not os.path.exists(local_path):
            load_model_from_azure(f"models/channel_{driver}.pt", local_path)

        if not os.path.exists(local_path):
            logging.warning(f"Skipping {driver} — no saved channel")
            continue

        df = load_driver_parquet(driver)
        if df is None or df.empty:
            logging.warning(f"Skipping {driver} — no data")
            continue

        ckpt  = torch.load(local_path, map_location=device)
        model = DriverChannel(
            input_size=ckpt["input_size"],
            radio_dim=ckpt["radio_dim"],
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        dataset = DriverDataset(df)
        if len(dataset) == 0:
            continue

        loader     = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        embs, tgts = [], []

        with torch.no_grad():
            for x_b, y_b, s_b, r_b, g_b in loader:
                emb, _ = model(
                    x_b.to(device),
                    s_b.to(device),
                    r_b.to(device),
                    g_b.to(device),
                )
                embs.append(emb.cpu().numpy())
                tgts.append(y_b.numpy())

        all_embeddings[driver] = np.concatenate(embs, axis=0)
        all_targets[driver]    = np.concatenate(tgts, axis=0)
        logging.info(
            f"{driver}: {all_embeddings[driver].shape[0]} embeddings extracted"
        )

    if not all_embeddings:
        raise ValueError(
            "No embeddings extracted — check gold container and saved channels"
        )

    min_n        = min(e.shape[0] for e in all_embeddings.values())
    first_driver = next(iter(all_targets))
    logging.info(f"Aligning RF matrix to {min_n} samples")

    X_rows, y_rows = [], []
    for i in range(min_n):
        row = [
            all_embeddings[d][i] if d in all_embeddings
            else np.zeros(EMBEDDING_DIM, dtype=np.float32)
            for d in DRIVER_LIST
        ]
        X_rows.append(np.concatenate(row))
        y_rows.append(all_targets[first_driver][i])

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int64)

    logging.info(f"RF input  : {X.shape}")
    logging.info(f"RF targets: {y.shape}")
    return X, y


# ── Train Random Forest ────────────────────────────────────────

def train_random_forest(
    save_dir: str,
    device:   torch.device,
) -> RandomForestClassifier:
    """
    Extract embeddings from all saved channels and train RF.
    Saves RF locally and uploads to platinum/models/random_forest.joblib
    """
    logging.info("Extracting embeddings for RF training...")
    X, y = extract_embeddings(save_dir, device)

    logging.info("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X, y)

    local_rf = os.path.join(save_dir, "random_forest.joblib")
    joblib.dump(rf, local_rf)
    logging.info(f"Random Forest saved locally: {local_rf}")

    save_model_to_azure(local_rf, "models/random_forest.joblib")

    return rf


# ── Load Channel ───────────────────────────────────────────────

def load_channel(
    driver:   str,
    save_dir: str,
    device:   torch.device,
) -> Optional[Tuple[DriverChannel, Dict]]:
    """
    Load a saved driver channel.
    Tries local first, downloads from platinum/models/ if missing.
    """
    local_path = os.path.join(save_dir, f"channel_{driver}.pt")

    if not os.path.exists(local_path):
        load_model_from_azure(f"models/channel_{driver}.pt", local_path)

    if not os.path.exists(local_path):
        logging.warning(f"No saved channel for {driver}")
        return None

    ckpt  = torch.load(local_path, map_location=device)
    model = DriverChannel(
        input_size=ckpt["input_size"],
        radio_dim=ckpt["radio_dim"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


# ── Inference ──────────────────────────────────────────────────

def predict(
    live_data: Dict[str, Dict],
    save_dir:  str,
    device:    torch.device,
) -> Dict[int, float]:
    """
    Run live inference for one race snapshot.

    live_data format:
    {
        "LEC": {
            "features":     np.ndarray  shape (SEQUENCE_LENGTH, input_size),
            "social_score": float,      NaN → filled with 5.0 automatically
            "radio":        np.ndarray  shape (SEQUENCE_LENGTH, radio_dim),
            "radio_gate":   np.ndarray  shape (SEQUENCE_LENGTH,)  0.0 or 1.0
        },
        "VER": { ... },
        ...
    }

    - Driver keys must match DRIVER_ABB keys (e.g. "LEC", "VER")
    - SEQUENCE_LENGTH = 300 timesteps = 30 seconds at 0.1s resolution
    - radio_gate comes from radio_data_available column
    - Missing drivers are zero-padded in the RF input

    Returns:
    {
        1:  0.42,   # P1 probability 5 minutes from now
        2:  0.31,
        ...
        20: 0.01,
    }
    """
    local_rf = os.path.join(save_dir, "random_forest.joblib")

    if not os.path.exists(local_rf):
        load_model_from_azure("models/random_forest.joblib", local_rf)

    if not os.path.exists(local_rf):
        raise FileNotFoundError(
            "Random Forest not found locally or in platinum/models/ on Azure."
        )

    rf  = joblib.load(local_rf)
    row = []

    for driver in DRIVER_LIST:
        if driver in live_data:
            result = load_channel(driver, save_dir, device)
            if result is not None:
                model, _ = result
                d = live_data[driver]

                x = torch.tensor(
                    d["features"], dtype=torch.float32
                ).unsqueeze(0).to(device)
                # (1, SEQUENCE_LENGTH, input_size)

                # Handle NaN social score
                social_val = d.get("social_score", 5.0)
                if social_val is None or (isinstance(social_val, float) and np.isnan(social_val)):
                    social_val = 5.0

                s = torch.tensor(
                    [social_val], dtype=torch.float32
                ).to(device)
                # (1,)

                r = torch.tensor(
                    d["radio"], dtype=torch.float32
                ).unsqueeze(0).to(device)
                # (1, SEQUENCE_LENGTH, radio_dim)

                g = torch.tensor(
                    d["radio_gate"], dtype=torch.float32
                ).unsqueeze(0).to(device)
                # (1, SEQUENCE_LENGTH)

                with torch.no_grad():
                    emb, _ = model(x, s, r, g)
                row.append(emb.cpu().numpy().squeeze())
            else:
                row.append(np.zeros(EMBEDDING_DIM, dtype=np.float32))
        else:
            row.append(np.zeros(EMBEDDING_DIM, dtype=np.float32))

    X       = np.concatenate(row).reshape(1, -1)
    probs   = rf.predict_proba(X)[0]
    classes = rf.classes_

    return {
        int(cls) + 1: round(float(p), 4)
        for cls, p in zip(classes, probs)
    }


# ── Entry Point ────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    SAVE_DIR = "./model_weights"
    DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Device             : {DEVICE}")
    logging.info(f"Session type       : {SESSION_TYPE}")
    logging.info(f"Sequence length    : {SEQUENCE_LENGTH} timesteps ({WINDOW_SECONDS}s)")
    logging.info(f"Prediction horizon : {PREDICTION_HORIZON} timesteps ({HORIZON_SECONDS}s / 5 min)")
    logging.info(f"Drivers            : {len(DRIVER_LIST)}")
    logging.info(f"Gold container     : {GOLD_CONTAINER}")
    logging.info(f"Model container    : {MODEL_CONTAINER}")
    logging.info(f"Model path pattern : {MODEL_CONTAINER}/models/channel_{{DRIVER}}.pt")

    # ── Step 1: Train all driver Bi-LSTM channels ───────────────
    logging.info("=" * 60)
    logging.info("STEP 1: Training Driver Channels")
    logging.info("=" * 60)
    train_all_channels(SAVE_DIR, DEVICE)

    # ── Step 2: Train Random Forest on frozen embeddings ────────
    logging.info("=" * 60)
    logging.info("STEP 2: Training Random Forest")
    logging.info("=" * 60)
    train_random_forest(SAVE_DIR, DEVICE)

    logging.info("=" * 60)
    logging.info("Training Complete")
    logging.info("=" * 60)
