#!/usr/bin/env bash

ENV_NAME="behavython"
APP_CMD="behavython-cli"

echo ""
echo "======================================="
echo "    Behavython CLI Interactive Menu"
echo "======================================="
echo ""
echo "1. DeepLabCut (Tracking and Filtering)"
echo "2. Analysis (Behavioral Metrics)"
echo ""
read -p "Select an option (1 or 2): " CHOICE

if [ "$CHOICE" == "1" ]; then
    echo ""
    echo "--- DeepLabCut Pipeline ---"
    read -p "Paste or drag video folder here: " INPUT_FOLDER
    INPUT_FOLDER="${INPUT_FOLDER%\"}"
    INPUT_FOLDER="${INPUT_FOLDER#\"}"
    INPUT_FOLDER="${INPUT_FOLDER%\'}"
    INPUT_FOLDER="${INPUT_FOLDER#\'}"

    read -p "Paste or drag DLC config.yaml here: " DLC_CONFIG
    DLC_CONFIG="${DLC_CONFIG%\"}"
    DLC_CONFIG="${DLC_CONFIG#\"}"
    DLC_CONFIG="${DLC_CONFIG%\'}"
    DLC_CONFIG="${DLC_CONFIG#\'}"

    read -p "Generate likelihood plots after tracking? (Y/N): " DO_PLOTS

    EXTRA_FLAGS="--run-tracking --dlc-config \"$DLC_CONFIG\""
    if [[ "$DO_PLOTS" =~ ^[Yy] ]]; then
        EXTRA_FLAGS="$EXTRA_FLAGS --likelihood-plots"
    fi

elif [ "$CHOICE" == "2" ]; then
    echo ""
    echo "--- Behavior Analysis Pipeline ---"
    read -p "Paste or drag analysis folder here: " INPUT_FOLDER
    INPUT_FOLDER="${INPUT_FOLDER%\"}"
    INPUT_FOLDER="${INPUT_FOLDER#\"}"
    INPUT_FOLDER="${INPUT_FOLDER%\'}"
    INPUT_FOLDER="${INPUT_FOLDER#\'}"

    echo ""
    echo "Experiment Types:"
    echo "1. open_field"
    echo "2. elevated_plus_maze"
    echo "3. y_maze"
    echo "4. radial_maze"
    echo "5. water_maze"
    echo "6. social_recognition"
    echo "7. social_discrimination"
    echo "8. njr"
    echo ""
    read -p "Select experiment type (1-8): " EXP_CHOICE

    case "$EXP_CHOICE" in
        1) EXP_TYPE="open_field" ;;
        2) EXP_TYPE="elevated_plus_maze" ;;
        3) EXP_TYPE="y_maze" ;;
        4) EXP_TYPE="radial_maze" ;;
        5) EXP_TYPE="water_maze" ;;
        6) EXP_TYPE="social_recognition" ;;
        7) EXP_TYPE="social_discrimination" ;;
        8) EXP_TYPE="njr" ;;
        *) echo "[ERROR] Invalid experiment type."; exit 1 ;;
    esac

    read -p "Paste or drag Arena JSON configuration (leave empty if none): " ARENA_CONFIG
    ARENA_CONFIG="${ARENA_CONFIG%\"}"
    ARENA_CONFIG="${ARENA_CONFIG#\"}"
    ARENA_CONFIG="${ARENA_CONFIG%\'}"
    ARENA_CONFIG="${ARENA_CONFIG#\'}"

    read -p "Generate output plots? (Y/N): " DO_PLOTS

    EXTRA_FLAGS="--experiment-type \"$EXP_TYPE\" --output \"$INPUT_FOLDER/cli_output\""
    if [ -n "$ARENA_CONFIG" ]; then
        EXTRA_FLAGS="$EXTRA_FLAGS --arena-config \"$ARENA_CONFIG\""
    fi
    if [[ "$DO_PLOTS" =~ ^[Nn] ]]; then
        EXTRA_FLAGS="$EXTRA_FLAGS --no-plots"
    fi

else
    echo "[ERROR] Invalid selection."
    exit 1
fi

if [ ! -d "$INPUT_FOLDER" ]; then
    echo "[ERROR] The folder \"$INPUT_FOLDER\" does not exist."
    exit 1
fi

# --- Find conda/mamba ---
echo ""
echo "[INFO] Searching for conda/mamba..."
CONDA_CMD=""

# Common install paths for Linux/Mac
PATHS=(
    "$HOME/miniforge3"
    "$HOME/mambaforge"
    "$HOME/miniconda3"
    "/opt/miniforge3"
    "/opt/miniconda3"
)

# If conda/mamba isn't found in paths, check if it's already in PATH
if command -v mamba >/dev/null 2>&1; then
    CONDA_CMD="mamba"
    echo "[INFO] Found mamba in PATH"
elif command -v conda >/dev/null 2>&1; then
    CONDA_CMD="conda"
    echo "[INFO] Found conda in PATH"
fi

if [ -z "$CONDA_CMD" ]; then
    for D in "${PATHS[@]}"; do
        if [ -x "$D/bin/mamba" ]; then
            CONDA_CMD="$D/bin/mamba"
            echo "[INFO] Found mamba at $D"
            break
        elif [ -x "$D/bin/conda" ]; then
            CONDA_CMD="$D/bin/conda"
            echo "[INFO] Found conda at $D"
            break
        fi
    done
fi

RUN_CMD=""

# --- If found, test env and set run command ---
if [ -n "$CONDA_CMD" ]; then
    echo "[INFO] Checking environment \"$ENV_NAME\"..."
    
    if "$CONDA_CMD" run -n "$ENV_NAME" python -c "import sys" >/dev/null 2>&1; then
        echo "[INFO] Environment found."
        RUN_CMD="$CONDA_CMD run --no-capture-output -n $ENV_NAME $APP_CMD"
    else
        echo "[WARN] Environment \"$ENV_NAME\" not found or invalid."
    fi
else
    echo "[WARN] No conda/mamba installation found."
fi

# --- Fallback ---
if [ -z "$RUN_CMD" ]; then
    echo "[INFO] Falling back to system environment..."
    RUN_CMD="$APP_CMD"
fi

# --- Check if CLI is available in the selected environment ---
echo "[INFO] Checking for $APP_CMD availability..."
if ! $RUN_CMD --help >/dev/null 2>&1; then
    echo "[ERROR] $APP_CMD could not be found or failed to run in this environment."
    echo "Please make sure Behavython is installed correctly."
    exit 1
fi

# --- Run Behavython CLI ---
echo "[INFO] Launching Behavython CLI..."
eval "$RUN_CMD --input-folder \"$INPUT_FOLDER\" $EXTRA_FLAGS"

echo ""
read -p "Press Enter to exit..."
