## Woad

A Vulkan based rasterizer/raytracer hybrid renderer.

This a simple exercise in exploring rendering possibilities with Vulkan, and is meant for personal learning.

### Build dependencies

A linux OS is required along with CMake and a C/C++ compiler. The library dependencies are quite minimal. You will need

* libxcb
* libxcb-keysyms
* libxcb-xinput

If you are missing any of these, they can likely be installed from your system package manager.

### Build instructions

I've included `make.sh` to simplify the build steps. It first unpacks a tarball, which contains assets and more source code, and then creates a build directory, changes to it, and invokes cmake.

I recommend taking a look at `make.sh` before running it to verify it is not doing anything evil.

After that, run the following:

```
git clone https://github.com/mogjira/woad
cd woad
bash make.sh
```

### Running

This is essentially a library, but there is one example binary included. After the build completes, from the project root directory, run:

```
./build/examples/ex00 --raytrace
```

That will run a simple demo. The `--raytrace` flag enables enables ray-tracing, which is currently only used for shadows.

I don't handle the case where ray-tracing is not available on the system. If it fails for that reason you can try omitting that flag. But, your shadows will not look as nice.

Once running, the following controls are available:

| Button | Action |
| ------ | ------ |
| Left Mouse | Tumble Camera |
| Middle Mouse | Pan Camera |
| Right Mouse | Zoom Camera |
| 'G' Key | Home Camera |
| 'Esc' Key | Terminate the demo |
