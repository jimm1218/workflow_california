# 自動化機器學習工作流 (Auto-ML Workflow) - 加州房價預測專案

本專案旨在利用一個標準化、可重用的 Python 工作流 (`workflow.py`)，對任何結構化的迴歸資料集進行快速、全面的機器學習分析。此次我們以经典的「加州房價資料集」(California Housing) 作為示範案例。

## 專案目標

過去在面對新的資料集時，常需要重複編寫 EDA、特徵工程、模型訓練等程式碼。為了解決此問題，我們建立了一個自動化的 CRISP-DM 工作流，期望達成以下目標：

1.  **一鍵執行**：僅需修改資料檔案、目標變數等幾個簡單參數，即可自動完成從資料探索到模型產出的完整流程。
2.  **標準化產出**：自動生成標準化的視覺化圖表、模型評估報告與可供部署的模型檔案。
3.  **可擴充性**：腳本採用物件導向設計，未來可輕易擴充新的模型、特徵工程方法或視覺化圖表。

## 自動化流程概覽

`workflow.py` 腳本遵循 CRISP-DM 方法論，自動執行以下步驟：

1.  **資料理解 (Data Understanding)**：自動讀取資料，並產出 EDA 綜合圖表，包含：
    *   數值特徵的相關性熱力圖。
    *   目標變數（房價中位數）的分佈直方圖。
    *   與房價最相關特徵的散佈圖。
    *   類別特徵對房價影響的箱型圖。
2.  **資料準備 (Data Preparation)**：
    *   自動偵測數值與類別特徵。
    *   對數值特徵進行遺失值填補（中位數）與標準化（StandardScaler）。
    *   對類別特徵進行獨熱編碼（One-Hot Encoding）。
3.  **模型建立 (Modeling)**：
    *   使用 5 種常見的迴歸模型進行訓練：`Linear Regression`, `Lasso`, `Ridge`, `Random Forest`, `Gradient Boosting`。
    *   針對每個模型，測試**所有可能的特徵組合**，找出最佳的 R² 與 RMSE 表現。
4.  **模型評估 (Evaluation)**：
    *   視覺化每個模型在不同特徵數量下的表現曲線。
    *   產出所有模型與特徵組合的總排行榜。
    *   透過多種特徵選擇演算法（F-Test, Mutual Information, RF Importance, Permutation Importance, Correlation）進行交叉驗證，並以熱力圖呈現共識結果。
5.  **模型部署 (Deployment)**：
    *   根據評估結果，自動選擇最佳的簡單模型（預設為線性迴歸的最佳特徵組合）。
    *   將包含「預處理」與「模型」的完整 Pipeline 封裝並保存為 `.joblib` 檔案，方便未來直接載入進行預測。

## 執行與產出

### 如何執行

確保 `California.csv` 與 `workflow.py` 在同一目錄下，然後執行：

```bash
python workflow.py
```

所有產出的圖表、報告與模型將會自動存放在 `output/` 資料夾中。

### 產出檔案列表

*   `eda_plots.png`: 探索性資料分析圖表。
*   `feature_combinations_*.png`: 各模型在不同特徵數下的表現。
*   `feature_combinations_top10_overall.png`: 綜合所有模型的最佳表現前 10 名。
*   `feature_selection_algorithms_metrics.png`: 多種特徵選擇演算法的共識熱力圖。
*   `feature_combinations_5models_metrics.csv`: 所有模型與特徵組合的詳細評估指標。
*   `feature_selection_metrics.csv`: 各特徵在不同選擇演算法下的重要性分數。
*   `full_pipeline_model.joblib`: 可直接用於預測的完整模型 Pipeline。
*   `preprocessor.joblib` / `multiple_linear_regression_model.joblib`: 分開保存的預處理器與模型。

## 分析結果摘要

### 1. 資料探索 (EDA)

從 EDA 圖表中，我們快速發現 `median_income`（收入中位數）與目標 `median_house_value`（房價中位數）有最強的正相關。同時，`ocean_proximity`（海洋鄰近度）這個類別特徵也顯著影響房價，靠近海灣（NEAR BAY）或海洋（NEAR OCEAN）的地區房價較高。

!EDA 探索性資料分析圖表

### 2. 模型與特徵組合分析

透過自動化的特徵組合測試，我們可以看到對於線性迴歸模型，並非特徵越多越好。大約在 8-9 個特徵時，模型的 R² 分數達到峰值，之後增加更多特徵反而可能因共線性等問題導致表現略微下降。

!線性迴歸特徵組合表現曲線

綜合所有模型的表現，`Gradient Boosting` 和 `Random Forest` 這類基於樹的模型，在使用較多特徵時表現最佳。

!綜合模型 TOP 10 排行榜

### 3. 特徵重要性共識

特徵選擇的共識熱力圖整合了五種不同演算法的觀點。顏色越深代表該特徵越重要。結果再次驗證了 `median_income` 的絕對重要性。此外，`ocean_proximity` 的幾個獨熱編碼後的類別（如 `<1H OCEAN`, `INLAND`）以及經緯度 `longitude` 和 `latitude` 也被多數演算法認為是關鍵特徵。

!特徵選擇共識熱力圖

### 4. 最終模型

工作流最終選擇了**線性迴歸模型**搭配其 R² 分數最高的特徵組合進行封裝。這個一體化的 `full_pipeline_model.joblib` 檔案，未來可以直接載入，輸入原始格式的新資料即可完成預處理與預測，極大地方便了模型的部署與應用。