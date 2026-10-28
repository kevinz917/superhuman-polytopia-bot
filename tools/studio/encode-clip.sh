#!/bin/zsh
# encode.sh <frames_dir> <out_basename> <stride> [width]
set -e
dir=$1; out=$2; stride=$3; width=${4:-600}
tmp=$(mktemp -d)
frames=($(python3 -c "import json,sys;l=json.load(open(sys.argv[1]+'/log.json'));m=max(e['frame'] for e in l);print(' '.join(e['file'] for e in l if 0<=e['frame']<m))" $dir))
i=0; n=0
for f in $frames; do
  if (( i % stride == 0 )); then ln -s $dir/$f $tmp/$(printf "s%05d.png" $n); n=$((n+1)); fi
  i=$((i+1))
done
last=$dir/${frames[-1]}
for k in $(seq 1 38); do ln -s $last $tmp/$(printf "s%05d.png" $n); n=$((n+1)); done
crop="crop=1400:900:200:90"
ffmpeg -loglevel error -y -framerate 15 -i $tmp/s%05d.png -vf "$crop,scale=$width:-1:flags=lanczos" \
  -c:v libwebp_anim -lossless 0 -q:v 72 -compression_level 6 -loop 0 $out.webp
ffmpeg -loglevel error -y -framerate 15 -i $tmp/s%05d.png -vf "$crop,scale=1400:-2:flags=lanczos" \
  -c:v libx264 -pix_fmt yuv420p -crf 20 -preset slow -movflags +faststart $out.mp4
rm -rf $tmp
ls -la $out.webp $out.mp4
