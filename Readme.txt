--- OpenCV instalation ---
Download OpenCV 4.5.0
Set OPENCV_DIR to "...\opencv\build"
Add to PATH "%OPENCV_DIR%\x64\vc15\bin" if VS 2017 or 2019 will be used or "%OPENCV_DIR%\x64\vc14\bin" if VS 2013(???)

--- More ---
Also install cmake and git

--- Building ---
For VS Code:
    Set toolchain to "[Visual Studio Community 201x Release - amd64]"
    Try using cmake build targets "Debug" and "Release"