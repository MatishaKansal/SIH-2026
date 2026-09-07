import os
import csv
from datasets import load_dataset, Audio
from huggingface_hub import login

# Default visible folder in your project workspace (not hidden)
DEFAULT_VISIBLE_CACHE = os.path.join(os.path.dirname(__file__), "hf_cache")
DEFAULT_EXPORT_DIR = os.path.join(os.path.dirname(__file__), "svarah_dataset")

def download_or_load_svarah(cache_dir=DEFAULT_VISIBLE_CACHE, token=None, streaming=False):
    """
    Load or download the ai4bharat/Svarah dataset into a visible project folder.
    
    :param cache_dir: Folder to store cache (defaults to ./data/hf_cache, a visible non-hidden directory)
    """
    if token:
        login(token=token)
    elif "HF_TOKEN" in os.environ:
        login(token=os.environ["HF_TOKEN"])
        
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Loading ai4bharat/Svarah dataset into visible cache: {cache_dir}")
    
    ds = load_dataset(
        "ai4bharat/Svarah", 
        cache_dir=cache_dir,
        streaming=streaming
    )
    print("Dataset loaded successfully!")
    return ds

def export_svarah_to_folder(output_dir=DEFAULT_EXPORT_DIR, limit=None):
    """
    Extracts and exports all .wav audio files and metadata.csv into a standard, 
    publicly visible folder structure.
    
    Output Structure:
      output_dir/
      ├── audio/
      │   ├── sample_1.wav
      │   └── ...
      └── metadata.csv
    """
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "metadata.csv")

    print(f"Fetching dataset...")
    ds = load_dataset("ai4bharat/Svarah", cache_dir=DEFAULT_VISIBLE_CACHE)
    
    # Cast column to avoid dependency decode issues and access raw WAV bytes directly
    raw_ds = ds["test"].cast_column("audio_filepath", Audio(decode=False))
    
    total = len(raw_ds) if limit is None else min(limit, len(raw_ds))
    print(f"Exporting {total} audio samples and metadata to: {output_dir}")

    fieldnames = [
        "filename", "duration", "text", "gender", "age_group", 
        "primary_language", "native_place_state", "native_place_district",
        "highest_qualification", "job_category", "occupation_domain"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, item in enumerate(raw_ds):
            if limit and idx >= limit:
                break

            audio_data = item["audio_filepath"]
            filename = audio_data.get("path", f"sample_{idx}.wav")
            wav_filepath = os.path.join(audio_dir, filename)

            # Save raw WAV file to disk
            if audio_data.get("bytes"):
                with open(wav_filepath, "wb") as f:
                    f.write(audio_data["bytes"])

            # Write record to CSV
            writer.writerow({
                "filename": filename,
                "duration": item.get("duration"),
                "text": item.get("text"),
                "gender": item.get("gender"),
                "age_group": item.get("age-group"),
                "primary_language": item.get("primary_language"),
                "native_place_state": item.get("native_place_state"),
                "native_place_district": item.get("native_place_district"),
                "highest_qualification": item.get("highest_qualification"),
                "job_category": item.get("job_category"),
                "occupation_domain": item.get("occupation_domain")
            })

            if (idx + 1) % 500 == 0 or (idx + 1) == total:
                print(f"Exported {idx + 1}/{total} samples...")

    print(f"\nExport Complete!")
    print(f"- Audio files directory: {audio_dir}")
    print(f"- Metadata CSV: {csv_path}")

if __name__ == "__main__":
    # Option 1: Load into a visible cache folder in your project workspace (./data/hf_cache)
    ds = download_or_load_svarah()
    
    # Option 2: Export all WAV files & metadata.csv into a standard readable folder (./data/svarah_dataset)
    # Uncomment the line below to run the export:
    # export_svarah_to_folder()