from pydub import AudioSegment
import os

# Paths
input_dir = "podcasts"
output_dir = "podcast_chunks"
chunk_length_ms = 30 * 1000  # 5 minutes

os.makedirs(output_dir, exist_ok=True)

def split_mp3(file_path, chunk_length_ms=chunk_length_ms):
    audio = AudioSegment.from_mp3(file_path)
    duration_ms = len(audio)
    basename = os.path.splitext(os.path.basename(file_path))[0]
    for i, start in enumerate(range(0, duration_ms, chunk_length_ms)):
        chunk = audio[start:start+chunk_length_ms]
        out_path = os.path.join(output_dir, f"{basename}_part{i+1}.mp3")
        chunk.export(out_path, format="mp3")
        print(f"Exported {out_path}")

for fname in os.listdir(input_dir):
    if fname.endswith(".mp3"):
        path = os.path.join(input_dir, fname)
        split_mp3(path)
