; Behavython Inno Setup 7 Installer Script
; High-DPI Modern Wizard Configuration with Pre-Install Checks and Dynamic Dark Theme

; ==============================================================================
; BUILD MODE SWITCH:
; Uncomment #define TEST_BUILD to build a 2-second UI test installer.
; Comment out #define TEST_BUILD to build the full production installer.
; #define TEST_BUILD

#define MyAppName "Behavython"
#define MyAppVersion "0.9.9rc11"
#define MyAppPublisher "Matheus Costa & João Pedro"
#define MyAppURL "https://github.com/mrdrzit/Behavython"
#define MyAppExeName "python.exe"

[Setup]
AppId={{8B3A8055-6F01-4D96-9D3E-BF876798C2C9}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
AppContact=matheuscosta3004@gmail.com
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile=..\LICENSE
InfoBeforeFile=info_before.txt
InfoAfterFile=info_after.txt
OutputDir=..\dist

#ifdef TEST_BUILD
OutputBaseFilename=Behavython-Setup-Test
#else
OutputBaseFilename=Behavython-Setup-{#MyAppVersion}
#endif

SetupIconFile=..\src\behavython\gui\assets\images\VY.ico
UninstallDisplayIcon={app}\VY.ico

; Modern Dynamic Theme (automatically adapts to Windows light/dark theme)
WizardStyle=modern dynamic
WizardImageAlphaFormat=defined
WizardSmallImageBackColor=none
WizardImageFile=wizard_side_100.bmp,wizard_side_200.bmp,wizard_side_250.bmp
WizardImageFileDynamicDark=wizard_side_100.bmp,wizard_side_200.bmp,wizard_side_250.bmp
WizardSmallImageFile=wizard_small_100.png,wizard_small_200.png,wizard_small_250.png
WizardSmallImageFileDynamicDark=wizard_small_100.png,wizard_small_200.png,wizard_small_250.png

Compression=lzma2/normal
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog commandline
DisableProgramGroupPage=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Messages]
WelcomeLabel1=Welcome to Behavython Setup
WelcomeLabel2=This wizard will install Behavython (PyTorch & DeepLabCut 3.0+ Edition) on your computer.%n%nAll dependencies, Python runtimes, and CUDA libraries are completely self-contained. No external Python or Conda installation is required.
ClickNext=Click Next to review hardware and version compatibility notes, or Cancel to exit Setup.
FinishedHeadingLabel=Completing the Behavython Setup Wizard
FinishedLabel=Behavython has been successfully installed on your computer.%n%nYou can launch the application from your Desktop shortcut or by searching "Behavython" in the Start Menu.

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
#ifdef TEST_BUILD
; Test stub files for rapid 2-second UI preview
Source: "..\build\test_dummy\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
#else
; Full production payload
Source: "..\build\staging_env\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
#endif
; Application icon
Source: "..\src\behavython\gui\assets\images\VY.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; Top-level Start Menu entry for Windows 11 All Apps and Search discovery
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Parameters: "-m behavython.main"; WorkingDir: "{app}"; IconFilename: "{app}\VY.ico"
; CLI entry point with help display and active prompt
Name: "{autoprograms}\{#MyAppName} CLI"; Filename: "{cmd}"; Parameters: "/K ""set PATH={app};{app}\Scripts;%PATH% && cd /d %USERPROFILE% && behavython-cli --help"""; WorkingDir: "{userdocs}"; IconFilename: "{app}\VY.ico"
; Program group entries
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Parameters: "-m behavython.main"; WorkingDir: "{app}"; IconFilename: "{app}\VY.ico"
Name: "{group}\{#MyAppName} CLI"; Filename: "{cmd}"; Parameters: "/K ""set PATH={app};{app}\Scripts;%PATH% && cd /d %USERPROFILE% && behavython-cli --help"""; WorkingDir: "{userdocs}"; IconFilename: "{app}\VY.ico"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
; Desktop shortcut (terminal attached)
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Parameters: "-m behavython.main"; WorkingDir: "{app}"; IconFilename: "{app}\VY.ico"; Tasks: desktopicon

[Run]
#ifndef TEST_BUILD
; Run prefix relocator on production installs
Filename: "{app}\Scripts\conda-unpack.exe"; StatusMsg: "Configuring Python environment and repairing paths (one-time setup)..."; Flags: runhidden
#endif
; Prompt user to launch application
Filename: "{app}\{#MyAppExeName}"; Parameters: "-m behavython.main"; Description: "{cm:LaunchProgram,{#MyAppName}}"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\Scripts\__pycache__"
Type: filesandordirs; Name: "{app}\Lib\site-packages\__pycache__"
Type: filesandordirs; Name: "{app}\__pycache__"
Type: filesandordirs; Name: "{app}\*.log"
Type: dirifempty; Name: "{app}"

[Code]
// Hardware pre-check: verify NVIDIA CUDA driver DLL exists on target system
function CheckNvidiaDriver(): Boolean;
begin
  Result := FileExists(ExpandConstant('{sys}\nvcuda.dll'));
end;

function InitializeSetup(): Boolean;
begin
  Result := True;
  if not CheckNvidiaDriver() then
  begin
    MsgBox('Hardware Notice: No NVIDIA GPU driver was detected on this computer (nvcuda.dll is missing).' + #13#10#13#10 +
           'Behavython will install and run normally in CPU fallback mode.' + #13#10 +
           'However, neural network tracking (DeepLabCut / SuperAnimal) will be significantly slower without GPU acceleration.' + #13#10#13#10 +
           'Minimum recommended: NVIDIA GeForce RTX 3060, RTX 2080, or higher with >= 12 GB VRAM and graphics driver >= 528.00.',
           mbInformation, MB_OK);
  end;
end;
