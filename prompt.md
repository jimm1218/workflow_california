# AI 輔助開發歷程 (Prompt Engineering)

本文件記錄了 `workflow.py` 這個自動化機器學習工作流的開發歷程。我們透過與 AI 程式碼助理的一系列對話（Prompts），從一個簡單的腳本逐步迭代，最終打造出一個功能強大且可重用的標準化流程。

---

### 初始 Prompt：建立基本框架

> **我 (User):**
> 請幫我寫一個 Python 腳本，為 `50_Startups.csv` 資料集建立一個基本的機器學習迴歸工作流。需求包含：
> 1.  讀取資料。
> 2.  定義特徵 (X) 與目標 (y)。
> 3.  將資料分割為訓練集與測試集。
> 4.  訓練一個簡單的線性迴歸模型。
> 5.  用 R² 分數評估模型表現。

**AI 產出:** 一個包含 `pandas`, `train_test_split`, `LinearRegression` 的基礎腳本。

---

### 第二階段：擴充與自動化

> **我 (User):**
> 很好！現在請幫我擴充這個腳本：
> 1.  將流程封裝成一個類 (Class)。
> 2.  加入更多模型，如 Lasso, Ridge, RandomForest, GradientBoosting，並用一個字典來管理。
> 3.  自動化 EDA 過程，幫我產生一個包含相關性熱力圖、目標分佈圖和散佈圖的綜合圖表並儲存。
> 4.  將所有產出的圖表和模型統一存放到 `./output` 資料夾。

**AI 產出:** 程式碼被重構成 `AutoCRISPWorkflow` 類別，增加了 `run_eda` 和 `run_modeling` 方法，並自動建立輸出資料夾。

---

### 第三階段：處理真實世界問題與錯誤

> **我 (User):**
> 我把資料更換成 `California.csv`，但執行時出現 `ValueError: Input X contains NaN`。請幫我修復。

**AI 產出:** 在預處理流程中加入了 `SimpleImputer(strategy='median')` 來自動填補數值特徵的遺失值。

> **我 (User):**
> 現在又出現 `FileNotFoundError`，因為腳本找不到 `California.csv`。

**AI 產出:** 修改了檔案讀取路徑，使用 `os.path.dirname(__file__)` 來建構絕對路徑，確保在任何位置執行腳本都能正確找到資料檔。

---

### 第四階段：進階分析與部署準備

> **我 (User):**
> 我想知道哪些特徵組合對模型最好。可以幫我寫一個迴圈來測試所有可能的特徵組合，並將結果視覺化，同時產出一個 Top 10 的排行榜嗎？

**AI 產出:** 引入 `itertools.combinations` 來生成特徵組合，並在模型訓練迴圈中迭代。新增了繪製每個模型表現曲線以及 Top 10 排行榜的輔助函式。

> **我 (User):**
> 除了模型表現，我還想從不同演算法的角度驗證特徵的重要性。請加入 F-Test, Mutual Information, 和 Permutation Importance，並用熱力圖呈現結果，讓我看看哪些特徵是真正的關鍵。

**AI 產出:** 新增 `run_feature_selection_analysis` 方法，整合多種 `sklearn` 的特徵選擇工具，並使用 `seaborn.heatmap` 將結果正規化後繪製成共識矩陣。

> **我 (User):**
> 我想把最終推薦的模型儲存起來以便未來使用，但遇到了 `PicklingError: Can't pickle local object`。

**AI 產出:** 解釋了錯誤原因（內部函式無法被序列化），並將 `Pipeline` 中使用的 `select_features` 函式移到類別外部成為全域函式，解決了 `joblib.dump` 的問題。同時，為了部署方便，額外保存了一個包含預處理和模型的完整 Pipeline 檔案。

---

### 最終階段：專案文檔化

> **我 (User):**
> 幫我生成一個 `README.md` 以及 `prompt.md`，內容主要是說明本次專案是透過過往 workflow 建立標準化自動流程，以及產出的結果說明並含圖。

**AI 產出:** 生成了您正在閱讀的這份 `prompt.md` 以及專案的 `README.md` 文件。