import pandas as pd
import numpy as np


def calculate_performance_metrics(df, ground_truth_col, prediction_col):
    """
    성능 지표 계산 함수

    Parameters:
    df: DataFrame
    ground_truth_col: Ground Truth 컬럼명 (True=정상, False=불량)
    prediction_col: Prediction 컬럼명 (True=정상, False=불량)
    """

    # 기본 통계
    total = len(df)

    # Ground Truth 통계
    gt_pass = (df[ground_truth_col] == True).sum()  # 실제 정상
    gt_fail = (df[ground_truth_col] == False).sum()  # 실제 불량

    # Prediction 통계
    pred_pass = (df[prediction_col] == True).sum()  # 예측 정상
    pred_fail = (df[prediction_col] == False).sum()  # 예측 불량

    # Confusion Matrix 계산
    # TP: 실제 정상(True)을 정상(True)으로 정확히 예측
    tp = ((df[ground_truth_col] == True) & (df[prediction_col] == True)).sum()

    # TN: 실제 불량(False)을 불량(False)으로 정확히 예측
    tn = ((df[ground_truth_col] == False) & (df[prediction_col] == False)).sum()

    # FP: 실제 불량(False)인데 정상(True)으로 잘못 예측
    fp = ((df[ground_truth_col] == False) & (df[prediction_col] == True)).sum()

    # FN: 실제 정상(True)인데 불량(False)으로 잘못 예측
    fn = ((df[ground_truth_col] == True) & (df[prediction_col] == False)).sum()

    # 성능 지표 계산
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (recall + specificity) / 2

    # 결과 딕셔너리
    results = {
        'Total': total,
        'Ground Truth - PASS': gt_pass,
        'Ground Truth - FAIL': gt_fail,
        'Prediction - PASS': pred_pass,
        'Prediction - FAIL': pred_fail,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1 Score': round(f1_score, 4),
        'Accuracy': round(accuracy, 4),
        'Specificity': round(specificity, 4),
        'Balanced Accuracy': round(balanced_accuracy, 4)
    }

    return results


if __name__ == '__main__':
    # 성능 지표 계산
    BASE_ROOTDIR = '../../eval'
    filename = f'{BASE_ROOTDIR}/merged_final_detail_results.xlsx'
    dataframe = pd.read_excel(filename)
    simple = calculate_performance_metrics(dataframe, 'GroundTruth', 'Predict')

    # 결과 출력
    print("=== 성능 평가 결과 ===")
    for key, value in simple.items():
        print(f"{key}: {value}")

    simple_df = pd.DataFrame([simple])
    simple_df.to_excel(f'{BASE_ROOTDIR}/kpi_metrics.xlsx', index=False)

    # 혼동 행렬 시각화
    print("\n=== Confusion Matrix ===")
    print(f"              Predicted")
    print(f"              FAIL  PASS")
    print(f"Actual FAIL   {simple['TP']:4d}  {simple['FN']:4d}")
    print(f"       PASS   {simple['FP']:4d}  {simple['TN']:4d}")
