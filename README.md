# SwarmUI

**SwarmUI v0.9.8 Beta**.

*Formerly known as StableSwarmUI.*

A Modular AI Image Generation Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility. Supports AI image models (*Krea 2, Stable Diffusion, Flux, etc.*), and AI video models (*MiniMax H3, Wan, LTX-2, etc.*), some AI audio models (ACE-Step, etc.) and will continue adding new ones in the future when they are released to the public.

![ui-screenshot](.github/images/swarmui.jpg)

- **Discord Community**: Join the [Discord](https://discord.gg/q2y38cqjNw) to discuss the project, get support, see announcements, etc.
- **Announcements**: Follow the [Feature Announcements Thread](https://github.com/mcmonkeyprojects/SwarmUI/discussions/1) for updates on new features.
- **General documentation**: [/docs folder](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/README.md)
- **Website**: [SwarmUI.net](https://swarmui.net/)

----


## MiniMax H3: One video, with or without reference pictures

**Making a video with H3, whether your prompt contains only text or also has attached pictures?** Select **MiniMax H3** as your main model in **Models**, then:

1. Leave **Image-to-Video Follow-up** **off**. Attaching even one picture to the prompt already makes the main H3 generation image-conditioned; no additional video model is needed.
2. Open **Main Video Generation** and set **Main Video Frames** to the length you want. This duration applies to H3 whether the prompt has zero, one, or several attached pictures.
3. Click **Generate**. You should get one video.

At the usual **24 FPS**, these frame settings give approximately:

| Video length | Main Video Frames |
| --- | ---: |
| 5 seconds | 124 |
| 10 seconds | 243 |
| 15 seconds | 362 |

H3 only supports certain frame counts, so the exact duration may vary slightly.

**Why the warning?** The old names "Text To Video" and "Image To Video" confused **what you provide as input** with **which generation stage runs**. They were not mutually exclusive modes. If H3 is both the main model and the follow-up video model, SwarmUI currently runs H3 twice: about 5 seconds with the base defaults, then another video from its first frame. Disabling the follow-up stage avoids the duplicate inference. It does **not** disable image references attached to your main H3 prompt.

**Want dedicated first-frame conditioning instead?** Select a regular *image* model as the main model, enable **Image-to-Video Follow-up**, and choose H3 FL2VA as **Video Model**. Upload your starting picture under **Init Image** and set **Init Image Creativity** to **0** to bypass the preliminary still-image sampler. Set the follow-up video length with **Image-to-Video Follow-up > Video Frames**. If you want SwarmUI to generate a new still image before animating it, leave the still-image generation stage enabled.

**You can also start an H3 video from a prompt-attached picture.** With H3 Ref2VA selected as the **main model**, attach several pictures directly to the prompt and describe the opening as `At 0.00 seconds, the video begins from <Picture 1>`. H3 often uses the first attached picture as the opening scene. This is **reference-guided** rather than a guaranteed, pixel-anchored first-frame keyframe. Keep **Image-to-Video Follow-up** off to avoid the unnecessary second video generation. **Init Image Creativity = 0** only controls preliminary still-image denoising in the separate **Init Image** workflow; it does not control prompt-reference conditioning.

For more video-model options, see [Video Model Support](docs/Video%20Model%20Support.md).

# Status

This project is in **Almost-Release** status. This means for most tasks, Swarm has excellent tooling available to you, but there are a few areas I still want to perfect. Swarm is recommended as an ideal UI for most users, beginners and pros alike.

Beginner users will love Swarm's primary Generate tab interface, making it easy to generate anything with a variety of powerful features. Advanced users may favor the Comfy Workflow tab to get the unrestricted raw graph, but will still have reason to come back to the Generate tab for convenience features (image editor, auto-workflow-generation, etc) and powertools (eg Grid Generator).

Those interested in helping push Swarm to a Full ready-for-anything perfected Release status are welcome to submit PRs (read the [Contributing](/CONTRIBUTING.md) document first), and you can contact us here on GitHub or on [Discord](https://discord.gg/q2y38cqjNw). I highly recommend reaching out to ask about plans for a feature before PRing it. There may already be specific plans or even a work in progress.

Key feature targets not yet implemented:
- LLM-assisted prompting (there's an extension for it, but LLM control should be natively supported)

# Donate

SwarmUI is 100% free and open source forever. If you want to help make sure it keeps pace with the best despite my refusal to paywall access or shove ads down your throat, [donate to SwarmUI](https://patreon.com/swarmui)!

# Try It On Google Colab

### Google Colab

**WARNING**: Google Colab does not necessarily allow remote WebUIs, particularly for free accounts, use at your own risk.

Colab link if you want to try Swarm: https://colab.research.google.com/github/mcmonkeyprojects/SwarmUI/blob/master/colab/colab-notebook.ipynb

# Run it on a Cloud GPU Provider

### Runpod

Runpod template (note: maintained by third party contributor [nerdylive123](https://github.com/nerdylive123)): https://get.runpod.io/swarmui

### Vast.ai

Vast.ai template ([readme](https://cloud.vast.ai/template/readme/8e5e6ab1fceb9db3f813e815907b3390)): https://cloud.vast.ai/?ref_id=62897&creator_id=62897&name=SwarmUI

Note it may take several minutes to start up the first time. Check the container logs to see setup progress. Check the template `?` info for hints on how to use.

# Installing on Windows

Note: if you're on Windows 10, you may need to manually install [git](https://git-scm.com/download/win) and [DotNET 8 SDK](https://dotnet.microsoft.com/en-us/download/dotnet/8.0) first. (Windows 11 this is automated).

- Download [The Install-Windows.bat file](https://github.com/mcmonkeyprojects/SwarmUI/releases/download/0.9.8-Beta/install-windows.bat), store it somewhere you want to install at (not `Program Files`), and run it.
    - It should open a command prompt and install itself.
    - If it closes without going further, try running it again, it sometimes needs to run twice. (TODO: Fix that)
    - It will place an icon on your desktop that you can use to re-launch the server at any time.
    - When the installer completes, it will automatically launch the SwarmUI server, and then after a moment show you the install UI.
    - Follow the install instructions on the page.
    - After you submit, be patient, some of the install processing take a few minutes (downloading models and etc).
    - See the [Basic Usage Guide](/docs/Basic%20Usage.md) to go from there.

(TODO): Even easier self-contained pre-installer, a `.msi` or `.exe` that provides a general install screen and lets you pick folder and all.

# Alternate Manual Windows Install

- Install git from https://git-scm.com/download/win
- Current version targets .NET 8, but a future version will target .NET 10, so install both:
- Install DotNET 8 SDK from https://dotnet.microsoft.com/en-us/download/dotnet/8.0 (Make sure to get the SDK x64 for Windows)
- Install DotNET 10 SDK from https://dotnet.microsoft.com/en-us/download/dotnet/10.0 (Make sure to get the SDK x64 for Windows)
- open a terminal to the folder you want swarm in and run `git clone https://github.com/mcmonkeyprojects/SwarmUI`
- open the folder and run `launch-windows.bat`

# Installing on Linux

### Prereqs

- Install `git` and `python3` via your OS package manager if they are not already installed (make sure to include `pip` and `venv` on distros that do not include them in python directly)
    - For example, on some Ubuntu (desktop) versions, `sudo apt install git python3-pip python3-venv`, or you may need <https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa>
    - For Debian or Ubuntu Server, `sudo apt install git python3-full`
    - You'll want Python 3.11 or 3.12. Things should also work fine with 3.10. 3.13 might work. Do not use 3.14 or later.
    - Make sure `python3.11 -m pip --version` returns a valid package

### Linux Easy Install

- Download [the install-linux.sh file](https://github.com/mcmonkeyprojects/SwarmUI/releases/download/0.6.5-Beta/install-linux.sh), store it somewhere you want to install at, and run it
    - If you like terminals, you can open a terminal to the folder and run the following commands: (Yes this link is still current):
        - `wget https://github.com/mcmonkeyprojects/SwarmUI/releases/download/0.6.5-Beta/install-linux.sh -O install-linux.sh`
        - `chmod +x install-linux.sh`
- Run the `./install-linux.sh` script, it will install everything for you and eventually open the webpage in your browser.
- Follow the install instructions on-page.
- See the [Basic Usage Guide](/docs/Basic%20Usage.md) to go from there.
- If you want to use Desktop app mode, it is not tested for Linux, but you probably need to install WPE Webkit: <https://wpewebkit.org/about/get-wpe.html#install-it-from-your-linux-distribution>

### Linux Manual Install

- Run shell commands:
    - `git clone https://github.com/mcmonkeyprojects/SwarmUI`
    - cd `SwarmUI`
- Current version targets .NET 8, but a future version will target .NET 10, so install both:
    - You can run shell command `./launchtools/linux-dotnet-install.sh`, or separately follow the instructions at:
        - <https://dotnet.microsoft.com/en-us/download/dotnet/8.0> and also <https://dotnet.microsoft.com/en-us/download/dotnet/10.0> (you need `dotnet-sdk-8.0`/`dotnet-sdk-10.0`, as that includes all relevant sub-packages)
- Open a shell terminal and `cd` to a directory you want to install into
- To launch, in the shell run:
    - `./launch-linux.sh`
    - or if running on a headless server, `./launch-linux.sh --launch_mode none --host 0.0.0.0` and/or swap host for [cloudflared](/docs/Advanced%20Usage.md)
- open `http://localhost:7801/Install` (if it doesn't launch itself)
- Follow the install instructions on-page.

### Linux Install Notes

- You can at any time in the future run the `launch-linux.sh` script to re-launch Swarm.
- If the page doesn't open itself, you can manually open `http://localhost:7801`

(TODO): Maybe outlink a dedicated document with per-distro details and whatever. Maybe also make a one-click installer for Linux? Can we remove the global python install prereq?

# Installing on Mac

> **Note**: You can only run SwarmUI on Mac computers with M-Series Apple silicon processors (eg M1, M2, ...).

- Open Terminal.
- Ensure your `brew` packages are updated with `brew update`.
- Verify your `brew` installation with `brew doctor`. You should not see any error in the command output.
- Install .NET for macOS: `brew install dotnet`.
- If you don't have Python, install it: `brew install python@3.11` and `brew install virtualenv`
    - Python 3.11, 3.12, 3.10 are all fine. 3.13 might work. Do not use 3.14 or later.
- Change the directory (`cd`) to the folder where you want to install SwarmUI.
- Clone the SwarmUI GitHub repository: `git clone https://github.com/mcmonkeyprojects/SwarmUI`.
- `cd SwarmUI` and run the installation script: `./launch-macos.sh`.
- Wait for the web browser to open, and follow the install instructions on-page.
- See the [Basic Usage Guide](/docs/Basic%20Usage.md) to go from there.

# Installing With Docker

See [Docs/Docker.md](/docs/Docker.md) for detailed instructions on using SwarmUI in Docker.

# Documentation

See [the documentation folder](/docs/README.md).

# Motivations

The "Swarm" name is in reference to the original key function of the UI: enabling a 'swarm' of GPUs to all generate images for the same user at once (especially for large grid generations). This is just the feature that inspired the name and not the end all of what Swarm is.

The overall goal of SwarmUI is to a be full-featured one-stop-shop for all things Stable Diffusion.

See [the motivations document](/docs/Motivations.md) for motivations on technical choices.

# Legal

This project:
- embeds a copy of [7-zip](https://7-zip.org/download.html) (LGPL).
- has the ability to auto-install [ComfyUI](https://github.com/comfyanonymous/ComfyUI) (GPL).
- has the option to use as a backend [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) (AGPL).
- can automatically install [christophschuhmann/improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) (Apache2) and [yuvalkirstain/PickScore](https://github.com/yuvalkirstain/PickScore) (MIT).
- can automatically install [git-for-windows](https://git-scm.com/download/win) (GPLv2).
- can automatically install MIT/BSD/Apache2/PythonSoftwareFoundation pip packages: [spandrel](https://pypi.org/project/spandrel/), [dill](https://pypi.org/project/dill/), [imageio-ffmpeg](https://pypi.org/project/imageio-ffmpeg/), [opencv-python-headless](https://pypi.org/project/opencv-python-headless/), [matplotlib](https://pypi.org/project/matplotlib/), [rembg](https://pypi.org/project/rembg/), [kornia](https://pypi.org/project/kornia/), [Cython](https://pypi.org/project/Cython/)
- can automatically install [ultralytics](https://github.com/ultralytics/ultralytics) (AGPL) for `YOLOv8` face detection (ie `SwarmYoloDetection` node or `<segment:yolo-...>` syntax usage may become subject to AGPL terms),
- can automatically install [insightface](https://github.com/deepinsight/insightface) (MIT) for `IP Adapter - Face` support
- uses [JSON.NET](https://github.com/JamesNK/Newtonsoft.Json) (MIT), [FreneticUtilities](https://github.com/FreneticLLC/FreneticUtilities) (MIT), [LiteDB](https://github.com/mbdavid/LiteDB) (MIT), [Magick.NET](https://github.com/dlemstra/Magick.NET) (Apache2), [ATL.NET](https://github.com/Zeugma440/atldotnet) (MIT)
- embeds copies of web assets from [BootStrap](https://getbootstrap.com/) (MIT), [Select2](https://select2.org/) (MIT), [JQuery](https://jquery.com/) (MIT), [exif-reader](https://github.com/mattiasw/ExifReader) (MPL-2.0), [music-metadata](https://github.com/Borewit/music-metadata) (MIT).
- contains some icons from [Cristian Munoz](https://www.figma.com/community/file/1311159026125960259/7000-free-ui-icons) (CC-BY-4.0), the font [inter by rsms](https://github.com/rsms/inter) (OFL), [Unifont by GNU](https://unifoundry.com/unifont/) (OFL), [Material Symbols Outlined by Google](https://fonts.google.com/icons) (Apache2).
- can be used to install some custom node packs, which have individual license notices for any non-pure-FOSS licenses before install.
- supports user-built extensions which may have their own licenses or legal conditions.

SwarmUI itself is under the MIT license, however some usages may be affected by the GPL variant licenses of connected projects list above, and note that any models used have their own licenses.

### Previous License

(For updates prior to June 2024)

The MIT License (MIT)
Copyright (c) 2024 Stability AI

### License

The MIT License (MIT)

Copyright (c) 2024-2026 Alex "mcmonkey" Goodwin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
