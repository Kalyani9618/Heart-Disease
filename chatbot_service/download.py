import os
from huggingface_hub import hf_hub_download

# Target folder
base_path = r"C:\Users\ggvfj\Downloads\Project\Heart\chatbot_service\models\medgemma"
os.makedirs(base_path, exist_ok=True)

print(f"--- Starting Download Process ---")
print(f"Target Folder: {base_path}")

try:
    # 1. Download the Main Model (Q5_K_M)
    print("Step 1: Downloading MedGemma 4B LLM...")
    llm_path = hf_hub_download(
        repo_id="unsloth/medgemma-4b-it-GGUF",
        filename="medgemma-4b-it-Q5_K_M.gguf",
        local_dir=base_path,
        local_dir_use_symlinks=False
    )
    print(f"Success: LLM saved at {llm_path}")

    # 2. Download the Vision Projector
    print("Step 2: Downloading Vision Projector...")
    vision_temp_path = hf_hub_download(
        repo_id="koboldcpp/mmproj",
        filename="gemma3-4b-mmproj.gguf",
        local_dir=base_path,
        local_dir_use_symlinks=False
    )

    # 3. Rename to your specific requirement
    final_vision_name = os.path.join(base_path, "mmproj-medgemma-4b-it-F16.gguf")
    if os.path.exists(vision_temp_path):
        # Remove old file if it exists to avoid rename errors
        if os.path.exists(final_vision_name):
            os.remove(final_vision_name)
        os.rename(vision_temp_path, final_vision_name)
        print(f"Success: Vision Projector renamed to {final_vision_name}")

    print("\n--- All downloads complete! ---")

except Exception as e:
    print(f"\nAn error occurred: {e}")