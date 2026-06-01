"""
download_expert_data.py

Downloads D4RL/pen/expert-v2 from Minari, filters successful episodes,
builds sliding-window action chunks, batch-normalizes observations, and
saves expert_data.pt: {"states": [N, 45], "action_chunks": [N, 4, 24]}
"""
import numpy as np
import torch
import minari

CHUNK_SIZE=4
EXPERT_DATA_PATH="expert_data.pt"
DATASET_ID="D4RL/pen/expert-v2"

def download_and_convert():
    print(f"Loading dataset:{DATASET_ID}")
    try:
        dataset=minari.load_dataset(DATASET_ID)
    except Exception:
        print("Not found, downloading...")
        minari.download_dataset(DATASET_ID)
        dataset=minari.load_dataset(DATASET_ID)

        print(f"  Total episodes : {dataset.total_episodes:,}")
        print(f"  Total steps    : {dataset.total_steps:,}")

    all_states_raw=[]
    all_chuncks=[]
    total_epi=0
    successful_epi=0
    total_transitions=0

    for episode in dataset.iterate_episodes():
        obs=np.array(episode.observations,dtype=np.float32)#[T+1,45] timestamp in one epi, differ for each epi
        actions=np.array(episode.actions,dtype=np.float32)#[T,24]
        T=len(actions)
        total_epi+=1

        #Adroit stores per step success in info
        success=False
        if episode.info is not None:
            for key in ("success","goal_achieved"):
                if key in episode.infos:
                    success=bool(np.any(episode.infos[key]))
                    break
        if not success:
            continue
        successful_epi+=1

        for i in range(T-CHUNK_SIZE+1):
            #all_states_raw:  [ [45], [45], [45], ... ]   length = N
            #all_chunks:      [ [4,24], [4,24], [4,24], ... ]   length = N (same N)
            all_states_raw.append(obs[i])
            all_chuncks.append(actions[i:i+CHUNK_SIZE])
            total_transitions+=1
        print(f"\n  Successful episodes : {successful_epi} / {total_epi}")
        print(f"  Transitions         : {total_transitions:,}")

    #fallback
    if total_transitions==0:
        print("\n  Warning: success info not found — using all episodes.")
        for episode in dataset.iterate_episodes():
            obs=np.array(episode.observations,dtype=np.float32)
            actions=np.array(episode.actions,dtype=np.float32)
            T=len(actions)
            for i in range(T-CHUNK_SIZE+1):
                all_states_raw.append(obs[i])
                all_chuncks.append(actions[i:i+CHUNK_SIZE])
                total_transitions+=1
        print(f"transitions all: {total_transitions:,}")

    #batch-normalize mean/std+tanh [-1,1]
    states_raw=np.array(all_states_raw,dype=np.float32)#n,45, n is the sum of  timestamp for each epi
    mean=states_raw.mean(axis=0)
    std=states_raw.std(axis=0)+1e-8
    states_norm=np.tanh((states_raw-mean)/std).astype(np.float32)

    chunk_np=np.array(all_chuncks,dtype=np.float32)#[N, 4, 24]
    states_t=torch.from_numpy(states_norm)
    chunks_t = torch.from_numpy(chunk_np)

    print(f"\nTensor shapes:")
    print(f"  states       : {states_t.shape}  dtype={states_t.dtype}")
    print(f"  action_chunks: {chunks_t.shape}  dtype={chunks_t.dtype}")

    torch.save({"states": states_t, "action_chunks": chunks_t}, EXPERT_DATA_PATH)
    print(f"\nSaved → {EXPERT_DATA_PATH}")


if __name__ == "__main__":
    download_and_convert()
