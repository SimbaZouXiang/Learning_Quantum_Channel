import sys
from TDME_Trott import Learning_TDME_scheduler

def main():
    print("Testing Learning_TDME_scheduler with num_threads=2")
    model, lloss, tloss, tloss_list = Learning_TDME_scheduler(
        N=4, 
        MPO_layer=1, 
        model_to_learn_layer=1, 
        mu=0.1, 
        gamma=[0.0, 0.0, 0.0, 0.0], 
        epochs=1,
        use_scheduler=False,
        num_threads=2
    )
    print("Testing finished successfully.")
    print(f"Test Loss: {tloss}")
    print(f"Number of test samples: {len(tloss_list)}")

if __name__ == "__main__":
    main()
