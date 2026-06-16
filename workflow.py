import os
import itertools
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.inspection import permutation_importance

import warnings
warnings.filterwarnings('ignore')

# 用於 Pipeline 的特徵篩選函式 (必須放在全域層級以避免 PicklingError)
def select_features_by_indices(X, indices):
    return X[:, indices]

class AutoCRISPWorkflow:
    """
    自動化 CRISP-DM 工作流
    能針對任意迴歸資料集自動執行 EDA、特徵工程、多模型訓練、特徵組合測試與特徵選擇驗證，並產出視覺化與模型檔案。
    """
    def __init__(self, data_path, target_col, output_dir='./output', test_size=0.2, random_state=42):
        self.data_path = data_path
        self.target_col = target_col
        self.output_dir = output_dir
        self.test_size = test_size
        self.random_state = random_state
        
        # 建立輸出目錄
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.df = None
        self.X = None
        self.y = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.preprocessor = None
        self.feature_names = []
        
        self.models = {
            'Linear_Regression': LinearRegression(),
            'Lasso_L1': Lasso(random_state=self.random_state),
            'Ridge_L2': Ridge(random_state=self.random_state),
            'Random_Forest': RandomForestRegressor(random_state=self.random_state),
            'Gradient_Boosting': GradientBoostingRegressor(random_state=self.random_state)
        }
        self.combination_results = []

    def execute(self):
        """一鍵執行完整流程"""
        print(f"🚀 開始執行自動化 CRISP-DM 流程: {self.data_path}")
        self.load_data()
        self.run_eda()
        self.prepare_data()
        self.run_modeling_combinations()
        self.run_feature_selection_analysis()
        self.train_and_save_final_model()
        print(f"✅ 流程執行完畢！所有產出已存入: {self.output_dir}")

    def load_data(self):
        """Phase 1: 讀取資料與基本定義"""
        self.df = pd.read_csv(self.data_path)
        self.X = self.df.drop(columns=[self.target_col])
        self.y = self.df[self.target_col]
        print(f"📊 資料載入成功 | 樣本數: {self.df.shape[0]} | 特徵數: {self.X.shape[1]}")

    def run_eda(self):
        """Phase 2: 探索性資料分析 (EDA)"""
        print("🔍 執行資料探索 (EDA)...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 相關性熱力圖 (數值特徵)
        numeric_df = self.df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=axes[0, 0], fmt=".2f")
        axes[0, 0].set_title("Correlation Heatmap")
        
        # 2. 目標變數分佈直方圖
        sns.histplot(self.df[self.target_col], kde=True, ax=axes[1, 1], color='purple')
        axes[1, 1].set_title(f"{self.target_col} Distribution")
        
        # 尋找相關性最高且作為 X 的特徵，用來畫散佈圖
        if len(numeric_df.columns) > 1:
            correlations = numeric_df.corr()[self.target_col].drop(self.target_col).abs()
            best_num_col = correlations.idxmax()
            
            # 檢查是否有類別特徵可用來當 hue
            cat_cols = self.X.select_dtypes(include=['object', 'category']).columns
            hue_col = cat_cols[0] if len(cat_cols) > 0 else None
            
            # 3. 散佈圖 (最強特徵 vs 目標)
            sns.scatterplot(data=self.df, x=best_num_col, y=self.target_col, hue=hue_col, ax=axes[0, 1])
            axes[0, 1].set_title(f"{best_num_col} vs {self.target_col}")
            
            # 4. 箱型圖 (類別特徵 vs 目標，若有的話)
            if hue_col:
                sns.boxplot(data=self.df, x=hue_col, y=self.target_col, ax=axes[1, 0])
                axes[1, 0].set_title(f"{self.target_col} by {hue_col}")
            else:
                # 如果沒有類別特徵，用來畫第二強的特徵散佈圖
                second_best_num_col = correlations.drop(best_num_col).idxmax()
                sns.scatterplot(data=self.df, x=second_best_num_col, y=self.target_col, ax=axes[1, 0])
                axes[1, 0].set_title(f"{second_best_num_col} vs {self.target_col}")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'eda_plots.png'))
        plt.close()
        print("   ✅ EDA 圖表已儲存 (eda_plots.png)")

    def prepare_data(self):
        """Phase 3: 資料準備與特徵工程"""
        print("⚙️ 執行特徵工程與資料分割...")
        numeric_features = self.X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = self.X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
            
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        
        # 擬合併轉換前處理器，以便提取特徵名稱
        self.preprocessor.fit(self.X_train)
        
        # 嘗試提取 OneHotEncoder 的特徵名稱
        cat_feature_names = []
        if categorical_features:
            cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_feature_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
            
        self.feature_names = numeric_features + cat_feature_names
        
        # 為了後續自由組合特徵，我們在這裡預先轉為 DataFrame
        self.X_train_proc = pd.DataFrame(self.preprocessor.transform(self.X_train), columns=self.feature_names)
        self.X_test_proc = pd.DataFrame(self.preprocessor.transform(self.X_test), columns=self.feature_names)
        print(f"   ✅ 前處理完成 | 最終特徵數: {len(self.feature_names)}")

    def run_modeling_combinations(self):
        """Phase 4: 多模型 × 特徵組合訓練"""
        print(f"🏋️ 開始模型訓練與特徵組合測試 (共 {len(self.models)} 個模型)...")
        
        # 產生所有可能的特徵組合 (如果特徵少於 10 個則窮舉，否則取重要的組合)
        all_combinations = []
        if len(self.feature_names) <= 10:
            for i in range(1, len(self.feature_names) + 1):
                for combo in itertools.combinations(self.feature_names, i):
                    all_combinations.append(list(combo))
        else:
            # 特徵過多時防呆，僅作逐次累加特徵測試
            all_combinations = [self.feature_names[:i] for i in range(1, len(self.feature_names) + 1)]
            
        print(f"   💡 將測試 {len(all_combinations)} 種特徵組合")
        
        for model_name, model in self.models.items():
            model_r2_scores = []
            model_rmse_scores = []
            feature_counts = []
            
            for combo in all_combinations:
                X_tr = self.X_train_proc[combo]
                X_te = self.X_test_proc[combo]
                
                model.fit(X_tr, self.y_train)
                preds = model.predict(X_te)
                
                r2 = r2_score(self.y_test, preds)
                rmse = np.sqrt(mean_squared_error(self.y_test, preds))
                
                self.combination_results.append({
                    'Model': model_name,
                    'Feature_Count': len(combo),
                    'Features': ', '.join(combo),
                    'R2': r2,
                    'RMSE': rmse
                })
                
                # 用於繪製單一模型曲線
                feature_counts.append(len(combo))
                model_r2_scores.append(r2)
                model_rmse_scores.append(rmse)
                
            # 繪製各模型的特徵組合曲線圖
            self._plot_model_performance(model_name, feature_counts, model_r2_scores, model_rmse_scores)
            
        # 儲存組合指標為 CSV
        results_df = pd.DataFrame(self.combination_results)
        results_df = results_df.sort_values(by='R2', ascending=False)
        results_df.to_csv(os.path.join(self.output_dir, 'feature_combinations_5models_metrics.csv'), index=False)
        
        # 產出全場排行榜 TOP 10 圖表
        self._plot_top_10_models(results_df)

    def _plot_model_performance(self, model_name, counts, r2_scores, rmse_scores):
        """輔助方法: 畫單一模型不同特徵數的表現"""
        df = pd.DataFrame({'Feature Count': counts, 'R2': r2_scores, 'RMSE': rmse_scores})
        # 若有重複特徵數量，取最好的 R2 作為代表線
        df_agg = df.groupby('Feature Count').agg({'R2':'max', 'RMSE':'min'}).reset_index()
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Number of Features')
        ax1.set_ylabel('R² Score', color=color)
        ax1.plot(df_agg['Feature Count'], df_agg['R2'], marker='o', color=color, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('RMSE', color=color)
        ax2.plot(df_agg['Feature Count'], df_agg['RMSE'], marker='x', color=color, linestyle='dashed')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title(f"{model_name} Performance vs Feature Count")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'feature_combinations_{model_name}.png'))
        plt.close()

    def _plot_top_10_models(self, results_df):
        """輔助方法: 畫 TOP 10 排行榜"""
        top_10 = results_df.head(10).copy()
        top_10['Label'] = top_10['Model'] + " (" + top_10['Feature_Count'].astype(str) + " feat)"
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=top_10, x='R2', y='Label', palette='viridis')
        plt.title("Top 10 Overall Models by R² Score")
        plt.xlabel("R² Score")
        plt.ylabel("Model Configuration")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_combinations_top10_overall.png'))
        plt.close()

    def run_feature_selection_analysis(self):
        """Phase 5: 特徵選擇演算法共識分析"""
        print("🧪 執行多種特徵選擇演算法驗證...")
        
        # 1. F-Test
        f_scores, _ = f_regression(self.X_train_proc, self.y_train)
        
        # 2. Mutual Information
        mi_scores = mutual_info_regression(self.X_train_proc, self.y_train, random_state=self.random_state)
        
        # 3. Random Forest Importance
        rf = RandomForestRegressor(random_state=self.random_state)
        rf.fit(self.X_train_proc, self.y_train)
        rf_importance = rf.feature_importances_
        
        # 4. Permutation Importance (on RF)
        perm_importance = permutation_importance(rf, self.X_test_proc, self.y_test, n_repeats=10, random_state=self.random_state)
        
        # 5. Correlation (Pearson)
        train_full = self.X_train_proc.copy()
        train_full['TARGET'] = self.y_train.values
        corr_scores = train_full.corr()['TARGET'].drop('TARGET').abs().values
        
        # 彙整結果
        fs_df = pd.DataFrame({
            'Feature': self.feature_names,
            'F-Test': f_scores / f_scores.max(), # 正規化以方便比較
            'Mutual Info': mi_scores / mi_scores.max(),
            'RF Importance': rf_importance / rf_importance.max(),
            'Permutation': perm_importance.importances_mean / perm_importance.importances_mean.max(),
            'Correlation': corr_scores / corr_scores.max()
        })
        
        fs_df.set_index('Feature', inplace=True)
        fs_df['Mean_Score'] = fs_df.mean(axis=1)
        fs_df = fs_df.sort_values(by='Mean_Score', ascending=False)
        
        # 畫圖
        plt.figure(figsize=(14, 8))
        sns.heatmap(fs_df.drop(columns=['Mean_Score']), annot=True, cmap='YlGnBu')
        plt.title("Feature Selection Algorithms Consensus (Normalized Scores)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_selection_algorithms_metrics.png'))
        plt.close()
        
        # 儲存評估結果
        fs_df.to_csv(os.path.join(self.output_dir, 'feature_selection_metrics.csv'))
        print("   ✅ 特徵選擇驗證完成 (feature_selection_algorithms_metrics.png)")

    def train_and_save_final_model(self):
        """Phase 6: 訓練最優的推薦模型 (此處預設採用 Linear Regression 配最佳單一/少數特徵)"""
        print("🎯 訓練與保存最終冠軍模型...")
        
        # 從組合結果中找出 Linear Regression 表現最好的簡單模型 (預設盡量少特徵)
        results_df = pd.DataFrame(self.combination_results)
        lr_results = results_df[results_df['Model'] == 'Linear_Regression'].copy()
        
        # 排序策略：R2最高，但如果R2極為接近，優先選特徵數最少的
        # 這裡簡化為直接取 R2 最高的特徵組合
        best_lr = lr_results.loc[lr_results['R2'].idxmax()]
        best_features = best_lr['Features'].split(', ')
        
        print(f"   🏆 最佳線性迴歸特徵組合: {best_features} (R²: {best_lr['R2']:.4f})")
        
        # 由於要封裝供 Production 使用，我們建立一個包含 Preprocessor 和最終特徵選擇的完整 Pipeline
        # 但因為我們前面已經處理過特徵，最安全的做法是保存整個 Pipeline
        
        # 自訂一個特徵篩選器
        from sklearn.preprocessing import FunctionTransformer
        # 在真實預測時，前處理完輸出 Numpy Array，需靠索引篩選
        # 透過全域函式與 kw_args 傳遞參數，成功避開 PicklingError
        indices = [self.feature_names.index(f) for f in best_features]
        feature_selector = FunctionTransformer(select_features_by_indices, kw_args={'indices': indices})
        
        final_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('feature_selector', feature_selector),
            ('regressor', LinearRegression())
        ])
        
        # 用全部訓練資料重新擬合 Pipeline
        final_pipeline.fit(self.X_train, self.y_train)
        
        # 驗證測試集表現
        final_preds = final_pipeline.predict(self.X_test)
        final_r2 = r2_score(self.y_test, final_preds)
        final_rmse = np.sqrt(mean_squared_error(self.y_test, final_preds))
        
        print(f"   📊 最終模型表現 - RMSE: {final_rmse:.2f} | R²: {final_r2:.4f}")
        
        # 儲存前處理器與模型 (分成兩個檔案，或如這邊示範直接存成一體化 Pipeline 更佳)
        model_loc = os.path.join(self.output_dir, 'multiple_linear_regression_model.joblib')
        joblib.dump(final_pipeline.named_steps['regressor'], model_loc)
        
        preprocessor_loc = os.path.join(self.output_dir, 'preprocessor.joblib')
        joblib.dump(self.preprocessor, preprocessor_loc)
        
        # 為了部署更方便，我們也把整個一體化 pipeline 存一份
        pipeline_loc = os.path.join(self.output_dir, 'full_pipeline_model.joblib')
        joblib.dump(final_pipeline, pipeline_loc)
        
        print("   💾 模型檔案儲存完畢 (.joblib)")

if __name__ == "__main__":
    # === 使用範例 ===
    # 只要將這段程式碼替換成未來的資料集參數即可一鍵產出報告
    
    # 獲取腳本所在的目錄，並建立絕對路徑，讓腳本在任何地方執行都能找到檔案
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    DATA_FILE = "California.csv" # 您的原始資料檔案名稱
    DATA_PATH = os.path.join(script_dir, DATA_FILE)
    TARGET_VARIABLE = "median_house_value"    # 請替換為目標預測變數欄位 (若欄位名稱不同請記得修改，例如 median_house_value)
    OUTPUT_FOLDER = os.path.join(script_dir, "output")    # 所有圖表與模型的存放路徑
    
    if os.path.exists(DATA_PATH):
        workflow = AutoCRISPWorkflow(
            data_path=DATA_PATH,
            target_col=TARGET_VARIABLE,
            output_dir=OUTPUT_FOLDER
        )
        workflow.execute()
    else:
        print(f"找不到資料檔案 '{DATA_PATH}'，請確認路徑。此腳本支援任何符合迴歸任務的結構化 CSV。")