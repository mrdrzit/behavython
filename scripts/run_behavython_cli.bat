@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ENV_NAME=behavython"
set "APP_CMD=python -m behavython.cli"
set "CONDA_CMD="
set "RUN_CMD="

:: ===========================================================================
:: Find conda/mamba and resolve run command (done once at startup)
:: ===========================================================================
echo [INFO] Searching for conda/mamba...

for %%D in (
    "%USERPROFILE%\miniforge3"
    "%USERPROFILE%\mambaforge"
    "%USERPROFILE%\mambaforge3"
    "%USERPROFILE%\miniconda3"
    "%ProgramData%\miniforge3"
    "%ProgramData%\mambaforge"
    "%ProgramData%\mambaforge3"
    "%ProgramData%\Miniconda3"
) do (
    if not defined CONDA_CMD (
        if exist "%%~D\Scripts\mamba.exe" (
            set "CONDA_CMD=%%~D\Scripts\mamba.exe"
            echo [INFO] Found mamba at %%~D
        ) else if exist "%%~D\condabin\conda.bat" (
            set "CONDA_CMD=%%~D\condabin\conda.bat"
            echo [INFO] Found conda at %%~D
        )
    )
)

if defined CONDA_CMD (
    call "%CONDA_CMD%" run -n "%ENV_NAME%" python -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "RUN_CMD=call "%CONDA_CMD%" run --no-capture-output -n "%ENV_NAME%" %APP_CMD%"
    )
)

if not defined RUN_CMD (
    set "RUN_CMD=%APP_CMD%"
)

%RUN_CMD% --help >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] %APP_CMD% could not be found or failed to run.
    echo Please make sure Behavython is installed correctly.
    pause
    exit /b 1
)

:: ===========================================================================
:: Main menu loop
:: ===========================================================================

:main_menu
cls
echo.
echo ============================================================
echo               Behavython CLI  ^|  Interactive Menu
echo ============================================================
echo.
echo   DeepLabCut
echo     1. Run Tracking
echo     2. Likelihood Plots
echo     3. Annotate Video
echo     4. Extract Frames
echo     5. Transfer Labels
echo.
echo   Pipeline
echo     6. Standardize Videos
echo     7. Crop Videos
echo     8. Cleanup Files
echo.
echo   Project
echo     9. Init (copy config templates)
echo    10. Analyze (behavioral metrics)
echo.
echo     0. Exit
echo.
set /p "CHOICE=Select an option: "

if "%CHOICE%"=="1"  goto :action_run_tracking
if "%CHOICE%"=="2"  goto :action_likelihood_plots
if "%CHOICE%"=="3"  goto :action_annotate_video
if "%CHOICE%"=="4"  goto :action_extract_frames
if "%CHOICE%"=="5"  goto :action_transfer_labels
if "%CHOICE%"=="6"  goto :action_standardize
if "%CHOICE%"=="7"  goto :action_crop_videos
if "%CHOICE%"=="8"  goto :action_cleanup
if "%CHOICE%"=="9"  goto :action_init
if "%CHOICE%"=="10" goto :action_analyze
if "%CHOICE%"=="0"  goto :eof

echo.
echo [ERROR] Invalid selection. Enter a number from 0 to 10.
pause
goto :main_menu


:: ===========================================================================
:: 1. Run Tracking
:: ===========================================================================

:action_run_tracking
cls
echo.
echo ---- Run Tracking ------------------------------------------
echo.

call :prompt_folder  "Video folder" "" INPUT_FOLDER
if "!INPUT_FOLDER!"=="_CANCELLED_" goto :main_menu

call :prompt_file    "DLC config.yaml" "" DLC_CONFIG ".yaml"
if "!DLC_CONFIG!"=="_CANCELLED_" goto :main_menu

call :prompt_yesno   "Generate likelihood plots after tracking? (Y/N)" "N" DO_PLOTS

set "FLAGS=--run-tracking --dlc-config "!DLC_CONFIG!""
if /i "!DO_PLOTS!"=="Y" set "FLAGS=!FLAGS! --likelihood-plots"

call :run_cli "!INPUT_FOLDER!" "!FLAGS!"
goto :main_menu


:: ===========================================================================
:: 2. Likelihood Plots
:: ===========================================================================

:action_likelihood_plots
cls
echo.
echo ---- Likelihood Plots --------------------------------------
echo.

call :prompt_folder  "Folder containing filtered .h5 files" "" INPUT_FOLDER
if "!INPUT_FOLDER!"=="_CANCELLED_" goto :main_menu

call :prompt_file    "DLC config.yaml" "" DLC_CONFIG ".yaml"
if "!DLC_CONFIG!"=="_CANCELLED_" goto :main_menu

set "FLAGS=--likelihood-plots --dlc-config "!DLC_CONFIG!""

call :run_cli "!INPUT_FOLDER!" "!FLAGS!"
goto :main_menu


:: ===========================================================================
:: 3. Annotate Video
:: ===========================================================================

:action_annotate_video
cls
echo.
echo ---- Annotate Video ----------------------------------------
echo.

call :prompt_folder  "Video folder" "" INPUT_FOLDER
if "!INPUT_FOLDER!"=="_CANCELLED_" goto :main_menu

call :prompt_file    "DLC config.yaml" "" DLC_CONFIG ".yaml"
if "!DLC_CONFIG!"=="_CANCELLED_" goto :main_menu

echo.
echo Output folder [leave empty for: !INPUT_FOLDER!\cli_output]:
set "OUTPUT_FOLDER="
set /p "OUTPUT_FOLDER=  > "
if defined OUTPUT_FOLDER set "OUTPUT_FOLDER=!OUTPUT_FOLDER:"=!"
if not defined OUTPUT_FOLDER set "OUTPUT_FOLDER=!INPUT_FOLDER!\cli_output"

set "FLAGS=--annotate-video --dlc-config "!DLC_CONFIG!" --output "!OUTPUT_FOLDER!""

call :run_cli "!INPUT_FOLDER!" "!FLAGS!"
goto :main_menu


:: ===========================================================================
:: 4. Extract Frames
:: ===========================================================================

:action_extract_frames
cls
echo.
echo ---- Extract Frames ----------------------------------------
echo.

call :prompt_folder  "Video folder" "" INPUT_FOLDER
if "!INPUT_FOLDER!"=="_CANCELLED_" goto :main_menu

echo.
echo Frame number override [leave empty for default: 50%% of video duration]:
set "FRAME_NUM="
set /p "FRAME_NUM=  > "
if defined FRAME_NUM set "FRAME_NUM=!FRAME_NUM:"=!"

if defined FRAME_NUM (
    echo !FRAME_NUM!| findstr /r "^[0-9][0-9]*$" >nul
    if errorlevel 1 (
        echo [ERROR] Frame number must be a positive integer.
        pause
        goto :action_extract_frames
    )
)

set "FLAGS=--extract-frames"
if defined FRAME_NUM set "FLAGS=!FLAGS! --frame-number !FRAME_NUM!"

call :run_cli "!INPUT_FOLDER!" "!FLAGS!"
goto :main_menu


:: ===========================================================================
:: 5. Transfer Labels
:: ===========================================================================

:action_transfer_labels
cls
echo.
echo ---- Transfer Labels ---------------------------------------
echo.

call :prompt_file    "Target DLC config.yaml (destination project)" "" DLC_CONFIG ".yaml"
if "!DLC_CONFIG!"=="_CANCELLED_" goto :main_menu

echo.
echo Source labeled-data folders (enter one per line, empty line when done):
set "LABEL_FOLDERS="
set "LF_COUNT=0"

:lf_loop
set "LF_INPUT="
set /p "LF_INPUT=  Folder !LF_COUNT! [empty to finish]: "
if defined LF_INPUT set "LF_INPUT=!LF_INPUT:"=!"
if not defined LF_INPUT goto :lf_done
if not exist "!LF_INPUT!" (
    echo [ERROR] Folder not found: !LF_INPUT!
    goto :lf_loop
)
set /a LF_COUNT+=1
set "LABEL_FOLDERS=!LABEL_FOLDERS! "!LF_INPUT!""
goto :lf_loop

:lf_done
if not defined LABEL_FOLDERS (
    echo [ERROR] At least one labeled-data folder is required.
    pause
    goto :action_transfer_labels
)

echo.
echo Output folder [leave empty for: same as source folders]:
set "OUTPUT_FOLDER="
set /p "OUTPUT_FOLDER=  > "
if defined OUTPUT_FOLDER set "OUTPUT_FOLDER=!OUTPUT_FOLDER:"=!"

set "FLAGS=--transfer-labels --dlc-config "!DLC_CONFIG!" --label-folders!LABEL_FOLDERS!"
if defined OUTPUT_FOLDER (
    if exist "!OUTPUT_FOLDER!" (
        set "FLAGS=!FLAGS! --output "!OUTPUT_FOLDER!""
    ) else (
        echo [WARN] Output folder does not exist, it will be created by the CLI.
        set "FLAGS=!FLAGS! --output "!OUTPUT_FOLDER!""
    )
)

:: Transfer Labels has no --input-folder; call CLI directly
echo.
echo Running: %RUN_CMD% !FLAGS!
echo.
%RUN_CMD% !FLAGS!
echo.
pause
goto :main_menu


:: ===========================================================================
:: 6. Standardize Videos
:: ===========================================================================

:action_standardize
cls
echo.
echo ---- Standardize Videos ------------------------------------
echo.

call :prompt_folder  "Video folder" "" INPUT_FOLDER
if "!INPUT_FOLDER!"=="_CANCELLED_" goto :main_menu

set "FLAGS=--standardize"

call :run_cli "!INPUT_FOLDER!" "!FLAGS!"
goto :main_menu


:: ===========================================================================
:: 7. Crop Videos
:: ===========================================================================

:action_crop_videos
cls
echo.
echo ---- Crop Videos -------------------------------------------
echo.

call :prompt_folder  "Video folder" "" INPUT_FOLDER
if "!INPUT_FOLDER!"=="_CANCELLED_" goto :main_menu

call :prompt_file    "Cropping project JSON" "" CROP_CONFIG ".json"
if "!CROP_CONFIG!"=="_CANCELLED_" goto :main_menu

set "FLAGS=--run-cropping --crop-config "!CROP_CONFIG!""

call :run_cli "!INPUT_FOLDER!" "!FLAGS!"
goto :main_menu


:: ===========================================================================
:: 8. Cleanup Files
:: ===========================================================================

:action_cleanup
cls
echo.
echo ---- Cleanup Files -----------------------------------------
echo.

call :prompt_folder  "Folder to clean" "" INPUT_FOLDER
if "!INPUT_FOLDER!"=="_CANCELLED_" goto :main_menu

echo.
echo Experiment type examples: open_field, elevated_plus_maze, y_maze,
echo   radial_maze, water_maze, social_recognition, social_discrimination, njr
echo.
set "EXP_TYPE="
set /p "EXP_TYPE=Experiment type: "
if defined EXP_TYPE set "EXP_TYPE=!EXP_TYPE:"=!"
if not defined EXP_TYPE (
    echo [ERROR] Experiment type is required for cleanup.
    pause
    goto :action_cleanup
)

set "FLAGS=--cleanup --experiment-type "!EXP_TYPE!""

call :run_cli "!INPUT_FOLDER!" "!FLAGS!"
goto :main_menu


:: ===========================================================================
:: 9. Init
:: ===========================================================================

:action_init
cls
echo.
echo ---- Init (copy config templates) --------------------------
echo.

echo Output folder [leave empty for current directory]:
set "OUTPUT_FOLDER="
set /p "OUTPUT_FOLDER=  > "
if defined OUTPUT_FOLDER set "OUTPUT_FOLDER=!OUTPUT_FOLDER:"=!"

echo.
echo Experiment type [leave empty to export all templates]:
echo   Examples: open_field, elevated_plus_maze
set "EXP_TYPE="
set /p "EXP_TYPE=  > "
if defined EXP_TYPE set "EXP_TYPE=!EXP_TYPE:"=!"

set "FLAGS=--init"
if defined OUTPUT_FOLDER (
    if not exist "!OUTPUT_FOLDER!" (
        echo [WARN] Output folder does not exist, it will be created.
    )
    set "FLAGS=!FLAGS! --output "!OUTPUT_FOLDER!""
)
if defined EXP_TYPE set "FLAGS=!FLAGS! --experiment-type "!EXP_TYPE!""

echo.
echo Running: %RUN_CMD% !FLAGS!
echo.
%RUN_CMD% !FLAGS!
echo.
pause
goto :main_menu


:: ===========================================================================
:: 10. Analyze
:: ===========================================================================

:action_analyze
cls
echo.
echo ---- Analyze (Behavioral Metrics) --------------------------
echo.

call :prompt_folder  "Analysis folder (CSVs, ROI files, etc.)" "" INPUT_FOLDER
if "!INPUT_FOLDER!"=="_CANCELLED_" goto :main_menu

echo.
echo Config file (JSON) [leave empty to use flags only]:
set "CONFIG_FILE="
set /p "CONFIG_FILE=  > "
if defined CONFIG_FILE set "CONFIG_FILE=!CONFIG_FILE:"=!"
if defined CONFIG_FILE (
    if not exist "!CONFIG_FILE!" (
        echo [ERROR] Config file not found: !CONFIG_FILE!
        pause
        goto :action_analyze
    )
)

if not defined CONFIG_FILE (
    echo.
    echo Experiment type examples: open_field, elevated_plus_maze, y_maze,
    echo   radial_maze, water_maze, social_recognition, social_discrimination, njr
    echo.
    set "EXP_TYPE="
    set /p "EXP_TYPE=Experiment type: "
    if defined EXP_TYPE set "EXP_TYPE=!EXP_TYPE:"=!"
    if not defined EXP_TYPE (
        echo [ERROR] Experiment type is required when no config file is provided.
        pause
        goto :action_analyze
    )
)

echo.
echo Arena config JSON [leave empty to skip]:
set "ARENA_CONFIG="
set /p "ARENA_CONFIG=  > "
if defined ARENA_CONFIG set "ARENA_CONFIG=!ARENA_CONFIG:"=!"
if defined ARENA_CONFIG (
    if not exist "!ARENA_CONFIG!" (
        echo [ERROR] Arena config file not found: !ARENA_CONFIG!
        pause
        goto :action_analyze
    )
)

echo.
echo Output folder [leave empty for: !INPUT_FOLDER!\cli_output]:
set "OUTPUT_FOLDER="
set /p "OUTPUT_FOLDER=  > "
if defined OUTPUT_FOLDER set "OUTPUT_FOLDER=!OUTPUT_FOLDER:"=!"
if not defined OUTPUT_FOLDER set "OUTPUT_FOLDER=!INPUT_FOLDER!\cli_output"

call :prompt_yesno   "Generate output plots? (Y/N)" "Y" DO_PLOTS

set "FLAGS="
if defined CONFIG_FILE   set "FLAGS=!FLAGS! --config "!CONFIG_FILE!""
if defined EXP_TYPE      set "FLAGS=!FLAGS! --experiment-type "!EXP_TYPE!""
if defined ARENA_CONFIG  set "FLAGS=!FLAGS! --arena-config "!ARENA_CONFIG!""
set "FLAGS=!FLAGS! --output "!OUTPUT_FOLDER!""
if /i "!DO_PLOTS!"=="N"  set "FLAGS=!FLAGS! --no-plots"

call :run_cli "!INPUT_FOLDER!" "!FLAGS!"
goto :main_menu


:: ===========================================================================
:: Subroutine: run_cli
::   %1 = input folder (quoted)
::   %2 = extra flags (pre-quoted where needed)
:: ===========================================================================

:run_cli
echo.
echo Running: %RUN_CMD% --input-folder %1 %~2
echo.
%RUN_CMD% --input-folder %1 %~2
echo.
pause
exit /b 0


:: ===========================================================================
:: Subroutine: prompt_folder
::   %1 = prompt label
::   %2 = default value (empty string = required)
::   %3 = output variable name
:: Sets output variable to _CANCELLED_ if user presses enter with no default.
:: ===========================================================================

:prompt_folder
set "_PF_LABEL=%~1"
set "_PF_DEFAULT=%~2"
set "_PF_VAR=%~3"

if defined _PF_DEFAULT (
    echo.
    echo !_PF_LABEL! [default: !_PF_DEFAULT!]:
) else (
    echo.
    echo !_PF_LABEL! ^(required^):
)

set "_PF_INPUT="
set /p "_PF_INPUT=  > "
if defined _PF_INPUT set "_PF_INPUT=!_PF_INPUT:"=!"

if not defined _PF_INPUT (
    if defined _PF_DEFAULT (
        set "!_PF_VAR!=!_PF_DEFAULT!"
        exit /b 0
    ) else (
        echo [ERROR] This field is required.
        pause
        goto :prompt_folder
    )
)

if not exist "!_PF_INPUT!" (
    echo [ERROR] Folder not found: !_PF_INPUT!
    pause
    set "_PF_INPUT="
    goto :prompt_folder
)

set "!_PF_VAR!=!_PF_INPUT!"
exit /b 0


:: ===========================================================================
:: Subroutine: prompt_file
::   %1 = prompt label
::   %2 = default value (empty = required)
::   %3 = output variable name
::   %4 = expected extension (e.g. .yaml), empty to skip check
:: ===========================================================================

:prompt_file
set "_PFL_LABEL=%~1"
set "_PFL_DEFAULT=%~2"
set "_PFL_VAR=%~3"
set "_PFL_EXT=%~4"

if defined _PFL_DEFAULT (
    echo.
    echo !_PFL_LABEL! [default: !_PFL_DEFAULT!]:
) else (
    echo.
    echo !_PFL_LABEL! ^(required^):
)

set "_PFL_INPUT="
set /p "_PFL_INPUT=  > "
if defined _PFL_INPUT set "_PFL_INPUT=!_PFL_INPUT:"=!"

if not defined _PFL_INPUT (
    if defined _PFL_DEFAULT (
        set "!_PFL_VAR!=!_PFL_DEFAULT!"
        exit /b 0
    ) else (
        echo [ERROR] This field is required.
        pause
        goto :prompt_file
    )
)

if not exist "!_PFL_INPUT!" (
    echo [ERROR] File not found: !_PFL_INPUT!
    pause
    set "_PFL_INPUT="
    goto :prompt_file
)

if defined _PFL_EXT (
    set "_PFL_ACTUAL_EXT=!_PFL_INPUT:*..=!"
    echo "!_PFL_INPUT!" | findstr /i /c:"!_PFL_EXT!" >nul
    if errorlevel 1 (
        echo [ERROR] Expected a !_PFL_EXT! file: !_PFL_INPUT!
        pause
        set "_PFL_INPUT="
        goto :prompt_file
    )
)

set "!_PFL_VAR!=!_PFL_INPUT!"
exit /b 0


:: ===========================================================================
:: Subroutine: prompt_yesno
::   %1 = prompt label
::   %2 = default (Y or N)
::   %3 = output variable name
:: ===========================================================================

:prompt_yesno
set "_PYN_LABEL=%~1"
set "_PYN_DEFAULT=%~2"
set "_PYN_VAR=%~3"

echo.
echo !_PYN_LABEL! [default: !_PYN_DEFAULT!]:
set "_PYN_INPUT="
set /p "_PYN_INPUT=  > "
if defined _PYN_INPUT set "_PYN_INPUT=!_PYN_INPUT:"=!"

if not defined _PYN_INPUT set "_PYN_INPUT=!_PYN_DEFAULT!"
set "_PYN_INPUT=!_PYN_INPUT: =!"

:: Validate input
if /i "!_PYN_INPUT!"=="Y" (
    set "!_PYN_VAR!=Y"
    exit /b 0
)
if /i "!_PYN_INPUT!"=="N" (
    set "!_PYN_VAR!=N"
    exit /b 0
)

echo [ERROR] Enter Y or N.
pause
goto :prompt_yesno

set "!_PYN_VAR!=!_PYN_INPUT!"
exit /b 0
