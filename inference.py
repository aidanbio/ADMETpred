import os
import argparse
import pandas as pd
import numpy as np
import torch

# 기존 모듈에서 필요한 함수들을 가져옵니다.
from eval_finetuned_tdc_2 import (
    load_model,
    predict_admet,
)

# evaluation.py에 정의된 22종 지표 리스트
FEATURE_NAMES = [
    "Caco2", "HIA", "Pgp", "Bioavailability", "Lipophilicity",
    "Solubility", "BBB", "PPBR", "VDss", "CYP2C9", "CYP2D6",
    "CYP3A4", "CYP2C9_Substrate", "CYP2D6_Substrate", "CYP3A4_Substrate",
    "Half_Life", "Clearance_Hepatocyte", "Clearance_Microsome",
    "LD50", "hERG", "AMES", "DILI",
]

def run_inference(smiles_list, output_dir, base_model_name, num_classes=22, batch_size=1):
    """
    SMILES 리스트에 대해 ADMET 22종 지표를 예측하고 결과를 반환합니다.
    """
    # 1. 체크포인트 및 설정 파일 경로 확인
    # 가장 최근 에폭이나 특정 체크포인트를 지정해야 합니다. (여기서는 예시로 checkpoint_epoch_1.pth 사용)
    ckpt_path = os.path.join(output_dir, "checkpoint_epoch_1.pth") 
    if not os.path.exists(ckpt_path):
        # 폴더 내에 있는 .pth 파일 중 하나를 자동으로 선택 (예비 로직)
        pts = [f for f in os.listdir(output_dir) if f.endswith(".pth")]
        if not pts:
            raise FileNotFoundError(f"No .pth checkpoint found in {output_dir}")
        ckpt_path = os.path.join(output_dir, pts[0])

    mean_std_path = os.path.join(output_dir, "label_mean_std.npz")

    print(f"[*] Loading model from {ckpt_path}...")
    
    # 2. 모델 로드
    model, tokenizer, mean_values, std_values, column_names = load_model(
        checkpoint_path=ckpt_path,
        mean_std_path=mean_std_path,
        num_classes=num_classes,
        base_model_name=base_model_name,
    )

    # 3. 예측 수행
    print(f"[*] Predicting for {len(smiles_list)} molecules...")
    # predict_admet은 (N, 22) 형태의 numpy array를 반환할 것으로 예상됩니다.
    preds = predict_admet(model, tokenizer, smiles_list, batch_size=batch_size)

    # 4. 정규화 해제 (Regression 지표의 경우)
    # evaluation.py의 로직을 참고하여, 표준화된 값을 원래 스케일로 복원합니다.
    # column_names 순서에 맞춰 mean/std를 적용해야 합니다.
    final_preds = preds.copy()
    for i, col in enumerate(column_names):
        if col in mean_values:
            final_preds[:, i] = (preds[:, i] * std_values[col]) + mean_values[col]

    # 5. 결과 정리 (DataFrame)
    results_df = pd.DataFrame(final_preds, columns=column_names)
    results_df.insert(0, "SMILES", smiles_list)
    
    return results_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", type=str, help="Single SMILES string to predict")
    parser.add_argument("--file", type=str, help="Path to a text file containing SMILES (one per line)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing model and mean_std.npz")
    parser.add_argument("--base_model", type=str, required=True, 
                        help="Base model name (e.g., 'sagawa/ZINC-deberta')")
    args = parser.parse_args()

    # 입력 소스 결정
    if args.smiles:
        input_smiles = [args.smiles]
    elif args.file:
        with open(args.file, "r") as f:
            input_smiles = [line.strip() for line in f if line.strip()]
    else:
        print("Error: Please provide --smiles or --file")
        return

    # 추론 실행
    try:
        results = run_inference(
            smiles_list=input_smiles,
            output_dir=args.output_dir,
            base_model_name=args.base_model
        )

        # 결과 출력
        print("\n" + "="*30)
        print("ADMET Inference Results")
        print("="*30)
        print(results.to_string(index=False))
        
        # CSV 저장 (선택 사항)
        # results.to_csv("inference_results.csv", index=False)
        
    except Exception as e:
        print(f"[-] Inference failed: {e}")

if __name__ == "__main__":
    main()
