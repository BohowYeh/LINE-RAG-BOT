{pkgs}: {
  deps = [
    pkgs.rustc
    pkgs.libiconv
    pkgs.cargo
    pkgs.bash
    pkgs.ffmpeg-full
  ];
}
