## Setup
The Labelme UI uses `QtPy`, which is **not** a Qt binding itself, but instead chooses an already-installed Qt binding such as `PyQt` or `PySide`.
In the main repo we use `PySide6`, but this does not work with this UI due to subtle differences. If you install `PyQt5` in a venv in this repository the UI should work.

## Config file
Default init config: `labelme/config/default_config.yaml`
Your custom config is created after running the program.  
Change your config by running (powershell / pycharm)`notepad $HOME\.labelmerc`
If there was an update to the default_config.yaml with new settings delete your local config to update it; `del $HOME\.labelmerc`

Same but for linux:\
`nano ~/.labelmerc`\
`rm ~/.labelmerc`

Please update the flag of your name to `true`  

## Shortcuts
(The shortcuts can be changed in the config too)

- Right, D: open_next
- Left, A: open_prev
- Up: select prev label
- Down: select next label
- Del: Delete point (First click 'edit poygons' then in the right column click the label to be deleted)

## Flags and label flags
Label flags are set for each new point.  
Flags are set for the whole image.  

You can add your own flags if useful in the configuration file.

## Options
- Next on click (default: true): Move to next image on click
- Delete similar on click (default: true): Delete similar labels on click


### Sam stuff
#### 1. Installation

Download the sam2 repo: https://github.com/facebookresearch/sam2  
Go to repo and do `pip install -e .`
#### 2. Use
After loaded one image, you can click `Edit` -> `Create Polygons with Segment Anything`. At the first time you use this option, it will freeze for a while and download the model weights(vit_h's weights file is around 2gb, so it could take some time).
After it downloaded, you can start annotating. Only points prompt is supported, you can left click to mark positive prompt and right click for negative prompt. When done, hit enter and it will generate polygons from the result.

Since we are editing in video mode, you should start at the frame that has some segmented object visible. All frames before that are not used.

#### 3. Configuration
The SAM related configuration is available in config file. You can find a `.labelmerc` file at user home directory after you open labelme for the first time (You need to delete old ones if you used official labelme previously).
Under sam section, you can adjust the settings accordingly.
```
sam:
  weights: vit_h # vit_h, vit_l or vit_b
  maxside: 1280 # will downsize the image during inference to the maxsize.
  approxpoly_epsilon: 1. # adjust the poylgon simplification algorithm. The larger the lesser vertices.
  device: "cuda" # "cuda" or "cpu"
```

#### Shortcuts SAM
-  Q: temporarily hide sam mask
-  Z: stop current sam
-  W: start SAM editing
-  S: compute SAM video
- Enter: save current mask to label
t the default is set to your name.
