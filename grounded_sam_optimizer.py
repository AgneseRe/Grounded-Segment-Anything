import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
import json

class ThresholdOptimizer:
    """
    Trova i threshold ottimali per GSAM attraverso grid search sistematico.
    """
    
    def __init__(
        self,
        labeler_class,
        root,
        img_dir,
        gt_dir,
        csv_path,
        gd_model,
        sam_predictor,
        device,
        sample_size: int = 200,
        output_dir: Path = Path("threshold_optimization")
    ):
        """
        Args:
            labeler_class: La classe GSAMDatasetLabeler
            sample_size: Numero di immagini da testare (200-500 consigliato)
            output_dir: Directory per salvare i risultati
        """
        self.labeler_class = labeler_class
        self.root = root
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.csv_path = csv_path
        self.gd_model = gd_model
        self.sam_predictor = sam_predictor
        self.device = device
        self.sample_size = sample_size
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Carica e campiona il dataset
        self.full_df = pd.read_csv(csv_path)
        self.sample_df = self._create_stratified_sample()
        
    def _create_stratified_sample(self) -> pd.DataFrame:
        """
        Crea un campione stratificato bilanciato per:
        - Numero di distractors (basso, medio, alto)
        - Tipo di target (se applicabile)
        """
        df = self.full_df.copy()
        
        # Stratifica per numero di distractors
        df['distractor_bin'] = pd.cut(
            df['num_distractors'], 
            bins=[0, 10, 30, float('inf')],
            labels=['low', 'medium', 'high']
        )
        
        # Campionamento stratificato
        sample_per_stratum = self.sample_size // 3
        sampled = df.groupby('distractor_bin', group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_per_stratum), random_state=42)
        )
        
        print(f"Campione creato: {len(sampled)} immagini")
        print(f"Distribuzione distractors:\n{sampled['distractor_bin'].value_counts()}")
        
        return sampled.reset_index(drop=True)
    
    def grid_search(
        self,
        box_thresholds: List[float] = [0.20, 0.23, 0.25, 0.27, 0.30, 0.33, 0.35],
        text_thresholds: List[float] = [0.20, 0.22, 0.25, 0.27, 0.30],
        nms_thresholds: List[float] = [0.40, 0.45, 0.50, 0.55, 0.60],
        iou_threshold: float = 0.75
    ) -> pd.DataFrame:
        """
        Esegue grid search completo su tutte le combinazioni di threshold.
        
        Returns:
            DataFrame con i risultati di ogni combinazione
        """
        results = []
        total_combinations = len(box_thresholds) * len(text_thresholds) * len(nms_thresholds)
        
        print(f"\n{'='*60}")
        print(f"GRID SEARCH: {total_combinations} combinazioni da testare")
        print(f"{'='*60}\n")
        
        with tqdm(total=total_combinations, desc="Grid Search Progress") as pbar:
            for box_t in box_thresholds:
                for text_t in text_thresholds:
                    for nms_t in nms_thresholds:
                        
                        # Testa questa combinazione
                        metrics = self._test_threshold_combination(
                            box_t, text_t, nms_t, iou_threshold
                        )
                        
                        results.append({
                            'box_threshold': box_t,
                            'text_threshold': text_t,
                            'nms_threshold': nms_t,
                            **metrics
                        })
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            'best_retention': f"{max([r['retention_rate'] for r in results]):.1f}%"
                        })
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / "grid_search_results.csv", index=False)
        
        return results_df
    
    def _test_threshold_combination(
        self,
        box_threshold: float,
        text_threshold: float,
        nms_threshold: float,
        iou_threshold: float
    ) -> Dict:
        """
        Testa una singola combinazione di threshold sul campione.
        """
        kept_count = 0
        total_iou_kept = 0
        total_iou_all = 0
        detection_failures = 0
        low_iou_count = 0
        iou_scores = []
        
        # Crea un labeler temporaneo con questi threshold
        temp_output = self.output_dir / "temp"
        temp_output.mkdir(exist_ok=True)
        
        labeler = self.labeler_class(
            root=self.root,
            img_dir=self.img_dir,
            gt_dir=self.gt_dir,
            csv_path=self.csv_path,
            out_dir=temp_output,
            gd_model=self.gd_model,
            sam_predictor=self.sam_predictor,
            device=self.device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            iou_threshold=iou_threshold,
            nms_threshold=nms_threshold,
            ada_box_threshold=False,
            ada_nms_threshold=False,
            max_images=None
        )
        
        # Simula il processing (versione leggera senza salvare files)
        for _, row in self.sample_df.iterrows():
            try:
                result = self._quick_process_image(labeler, row, iou_threshold)
                
                if result is None:  # Detection failure
                    detection_failures += 1
                    continue
                
                best_iou = result['best_iou']
                iou_scores.append(best_iou)
                total_iou_all += best_iou
                
                if best_iou >= iou_threshold:
                    kept_count += 1
                    total_iou_kept += best_iou
                else:
                    low_iou_count += 1
                    
            except Exception:
                detection_failures += 1
                continue
        
        # Calcola metriche
        total_processed = len(self.sample_df) - detection_failures
        retention_rate = (kept_count / len(self.sample_df) * 100) if len(self.sample_df) > 0 else 0
        avg_iou_kept = (total_iou_kept / kept_count) if kept_count > 0 else 0
        avg_iou_all = (total_iou_all / total_processed) if total_processed > 0 else 0
        
        return {
            'retention_rate': retention_rate,
            'kept_count': kept_count,
            'discarded_count': low_iou_count,
            'detection_failures': detection_failures,
            'avg_iou_kept': avg_iou_kept,
            'avg_iou_all': avg_iou_all,
            'median_iou': np.median(iou_scores) if iou_scores else 0,
            'std_iou': np.std(iou_scores) if iou_scores else 0,
            'min_iou': min(iou_scores) if iou_scores else 0,
            'max_iou': max(iou_scores) if iou_scores else 0
        }
    
    def _quick_process_image(self, labeler, row, iou_threshold):
        """
        Versione veloce di process_single_image che ritorna solo le metriche.
        """
        from PIL import Image
        from GroundingDINO.groundingdino.util.inference import load_image, predict, apply_nms
        from GroundingDINO.groundingdino.util import box_ops
        from grounded_sam_labeler_util import load_gt_mask, compute_iou
        
        image_name = row['image_name']
        class_name = row['target_type']
        
        # Load image
        image_path = labeler.img_dir / image_name
        if not image_path.exists():
            return None
        
        image, image_transformed = load_image(str(image_path))
        height, width, _ = image.shape
        
        # Run Grounding DINO
        boxes, logits, phrases = predict(
            model=labeler.gd_model,
            image=image_transformed,
            caption=class_name,
            box_threshold=labeler.box_threshold,
            text_threshold=labeler.text_threshold,
            nms_threshold=None
        )
        
        if boxes is None or len(boxes) == 0:
            return None
        
        # Apply NMS
        if labeler.nms_threshold is not None:
            boxes, logits, phrases = apply_nms(boxes, logits, phrases, labeler.nms_threshold)
        
        # Load GT
        gt_path = load_gt_mask(labeler.gt_dir, image_name)
        if gt_path is None:
            return None
        
        gt_mask = Image.open(gt_path).convert("L")
        gt_mask_np = np.array(gt_mask)
        gt_mask_bin = (gt_mask_np > 127).astype(np.uint8)
        
        # Run SAM
        labeler.sam_predictor.set_image(image)
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * labeler.device.new_tensor([width, height, width, height])
        transformed_boxes = labeler.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image.shape[:2]).to(labeler.device)
        
        masks, _, _ = labeler.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        
        # Compute IoUs
        ious = []
        for mask_tensor in masks:
            mask_np = mask_tensor[0].detach().cpu().numpy()
            iou = compute_iou(mask_np, gt_mask_bin)
            ious.append(iou)
        
        best_iou = max(ious) if ious else 0.0
        
        return {
            'best_iou': best_iou,
            'num_masks': len(ious)
        }
    
    def analyze_results(self, results_df: pd.DataFrame):
        """
        Analizza i risultati e genera visualizzazioni.
        """
        print(f"\n{'='*60}")
        print("ANALISI RISULTATI")
        print(f"{'='*60}\n")
        
        # Top 10 combinazioni
        top_10 = results_df.nlargest(10, 'retention_rate')
        print("Top 10 Combinazioni per Retention Rate:")
        print(top_10[['box_threshold', 'text_threshold', 'nms_threshold', 
                      'retention_rate', 'avg_iou_kept']].to_string(index=False))
        
        # Migliore combinazione
        best = results_df.loc[results_df['retention_rate'].idxmax()]
        print(f"\n{'='*60}")
        print("MIGLIORE COMBINAZIONE:")
        print(f"{'='*60}")
        print(f"Box Threshold: {best['box_threshold']:.2f}")
        print(f"Text Threshold: {best['text_threshold']:.2f}")
        print(f"NMS Threshold: {best['nms_threshold']:.2f}")
        print(f"Retention Rate: {best['retention_rate']:.2f}%")
        print(f"Avg IoU (kept): {best['avg_iou_kept']:.3f}")
        print(f"Detection Failures: {best['detection_failures']}")
        
        # Salva la migliore configurazione
        best_config = {
            'box_threshold': float(best['box_threshold']),
            'text_threshold': float(best['text_threshold']),
            'nms_threshold': float(best['nms_threshold']),
            'expected_retention_rate': float(best['retention_rate']),
            'avg_iou_kept': float(best['avg_iou_kept'])
        }
        
        with open(self.output_dir / "best_config.json", 'w') as f:
            json.dump(best_config, f, indent=4)
        
        print(f"\nConfigurazione salvata in: {self.output_dir / 'best_config.json'}")
        
        # Genera visualizzazioni
        self._plot_heatmaps(results_df)
        self._plot_pareto_frontier(results_df)
        self._plot_threshold_sensitivity(results_df)
        
        return best_config
    
    def _plot_heatmaps(self, results_df: pd.DataFrame):
        """
        Crea heatmap per visualizzare l'impatto dei threshold.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Heatmap 1: Box vs Text threshold (media su NMS)
        pivot1 = results_df.groupby(['box_threshold', 'text_threshold'])['retention_rate'].mean().reset_index()
        pivot1_matrix = pivot1.pivot(index='text_threshold', columns='box_threshold', values='retention_rate')
        
        sns.heatmap(pivot1_matrix, annot=True, fmt='.1f', cmap='RdYlGn', ax=axes[0, 0], 
                    vmin=0, vmax=100, cbar_kws={'label': 'Retention Rate (%)'})
        axes[0, 0].set_title('Box vs Text Threshold (avg over NMS)', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Text Threshold')
        axes[0, 0].set_xlabel('Box Threshold')
        
        # Heatmap 2: Box vs NMS threshold (media su Text)
        pivot2 = results_df.groupby(['box_threshold', 'nms_threshold'])['retention_rate'].mean().reset_index()
        pivot2_matrix = pivot2.pivot(index='nms_threshold', columns='box_threshold', values='retention_rate')
        
        sns.heatmap(pivot2_matrix, annot=True, fmt='.1f', cmap='RdYlGn', ax=axes[0, 1],
                    vmin=0, vmax=100, cbar_kws={'label': 'Retention Rate (%)'})
        axes[0, 1].set_title('Box vs NMS Threshold (avg over Text)', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('NMS Threshold')
        axes[0, 1].set_xlabel('Box Threshold')
        
        # Heatmap 3: Text vs NMS threshold (media su Box)
        pivot3 = results_df.groupby(['text_threshold', 'nms_threshold'])['retention_rate'].mean().reset_index()
        pivot3_matrix = pivot3.pivot(index='nms_threshold', columns='text_threshold', values='retention_rate')
        
        sns.heatmap(pivot3_matrix, annot=True, fmt='.1f', cmap='RdYlGn', ax=axes[1, 0],
                    vmin=0, vmax=100, cbar_kws={'label': 'Retention Rate (%)'})
        axes[1, 0].set_title('Text vs NMS Threshold (avg over Box)', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('NMS Threshold')
        axes[1, 0].set_xlabel('Text Threshold')
        
        # Heatmap 4: Avg IoU (kept images)
        pivot4 = results_df.groupby(['box_threshold', 'text_threshold'])['avg_iou_kept'].mean().reset_index()
        pivot4_matrix = pivot4.pivot(index='text_threshold', columns='box_threshold', values='avg_iou_kept')
        
        sns.heatmap(pivot4_matrix, annot=True, fmt='.3f', cmap='viridis', ax=axes[1, 1],
                    cbar_kws={'label': 'Avg IoU (kept)'})
        axes[1, 1].set_title('Average IoU of Kept Images', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Text Threshold')
        axes[1, 1].set_xlabel('Box Threshold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "threshold_heatmaps.png", dpi=300, bbox_inches='tight')
        print(f"Heatmaps salvate in: {self.output_dir / 'threshold_heatmaps.png'}")
        plt.close()
    
    def _plot_pareto_frontier(self, results_df: pd.DataFrame):
        """
        Plotta la frontiera di Pareto: retention rate vs qualità (avg IoU).
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        scatter = ax.scatter(
            results_df['retention_rate'],
            results_df['avg_iou_kept'],
            c=results_df['detection_failures'],
            s=100,
            alpha=0.6,
            cmap='coolwarm',
            edgecolors='black',
            linewidth=0.5
        )
        
        # Identifica punti Pareto-ottimali
        pareto_mask = self._is_pareto_efficient(
            results_df[['retention_rate', 'avg_iou_kept']].values
        )
        pareto_points = results_df[pareto_mask]
        
        ax.scatter(
            pareto_points['retention_rate'],
            pareto_points['avg_iou_kept'],
            s=200,
            marker='*',
            c='gold',
            edgecolors='black',
            linewidth=2,
            label='Pareto Optimal',
            zorder=10
        )
        
        # Annotate best point
        best = results_df.loc[results_df['retention_rate'].idxmax()]
        ax.annotate(
            f"Best\n({best['retention_rate']:.1f}%, {best['avg_iou_kept']:.3f})",
            xy=(best['retention_rate'], best['avg_iou_kept']),
            xytext=(10, 10),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2)
        )
        
        ax.set_xlabel('Retention Rate (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average IoU (kept images)', fontsize=12, fontweight='bold')
        ax.set_title('Pareto Frontier: Retention vs Quality', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Detection Failures', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "pareto_frontier.png", dpi=300, bbox_inches='tight')
        print(f"Pareto frontier salvata in: {self.output_dir / 'pareto_frontier.png'}")
        plt.close()
    
    def _plot_threshold_sensitivity(self, results_df: pd.DataFrame):
        """
        Analisi di sensitività: come ogni threshold impatta la retention.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        thresholds = ['box_threshold', 'text_threshold', 'nms_threshold']
        titles = ['Box Threshold', 'Text Threshold', 'NMS Threshold']
        
        for i, (thresh, title) in enumerate(zip(thresholds, titles)):
            grouped = results_df.groupby(thresh).agg({
                'retention_rate': ['mean', 'std', 'min', 'max'],
                'avg_iou_kept': 'mean'
            }).reset_index()
            
            ax = axes[i]
            x = grouped[thresh]
            y_mean = grouped[('retention_rate', 'mean')]
            y_std = grouped[('retention_rate', 'std')]
            
            ax.plot(x, y_mean, 'o-', linewidth=2, markersize=8, label='Mean Retention')
            ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3, label='±1 std')
            
            ax2 = ax.twinx()
            ax2.plot(x, grouped[('avg_iou_kept', 'mean')], 's--', color='red', 
                     linewidth=2, markersize=6, label='Mean IoU (kept)')
            
            ax.set_xlabel(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('Retention Rate (%)', fontsize=11)
            ax2.set_ylabel('Avg IoU (kept)', fontsize=11, color='red')
            ax.set_title(f'Impact of {title}', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "threshold_sensitivity.png", dpi=300, bbox_inches='tight')
        print(f"Sensitivity analysis salvata in: {self.output_dir / 'threshold_sensitivity.png'}")
        plt.close()
    
    def _is_pareto_efficient(self, costs):
        """
        Trova i punti Pareto-efficienti (massimizza entrambe le dimensioni).
        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)
                is_efficient[i] = True
        return is_efficient


# ========== ESEMPIO DI UTILIZZO ==========

def run_optimization(
    labeler_class,
    root,
    img_dir,
    gt_dir,
    csv_path,
    gd_model,
    sam_predictor,
    device
):
    """
    Esegue l'ottimizzazione dei threshold.
    """
    
    optimizer = ThresholdOptimizer(
        labeler_class=labeler_class,
        root=root,
        img_dir=img_dir,
        gt_dir=gt_dir,
        csv_path=csv_path,
        gd_model=gd_model,
        sam_predictor=sam_predictor,
        device=device,
        sample_size=200,  # Usa 200 immagini per il test
        output_dir=Path("threshold_optimization")
    )
    
    # Esegui grid search
    results_df = optimizer.grid_search(
        box_thresholds=[0.20, 0.23, 0.25, 0.27, 0.30, 0.33],
        text_thresholds=[0.20, 0.22, 0.25, 0.27, 0.30],
        nms_thresholds=[0.40, 0.45, 0.50, 0.55, 0.60],
        iou_threshold=0.75
    )
    
    # Analizza i risultati
    best_config = optimizer.analyze_results(results_df)
    
    print("\n" + "="*60)
    print("OTTIMIZZAZIONE COMPLETATA!")
    print("="*60)
    print("\nUsa questa configurazione nel tuo labeler:")
    print(f"  box_threshold = {best_config['box_threshold']}")
    print(f"  text_threshold = {best_config['text_threshold']}")
    print(f"  nms_threshold = {best_config['nms_threshold']}")
    print(f"\nRetention attesa: {best_config['expected_retention_rate']:.2f}%")
    
    return best_config

# Nel tuo script principale:
# best_config = run_optimization(
#     labeler_class=GSAMDatasetLabeler,
#     root=root_dir,
#     img_dir=img_dir,
#     gt_dir=gt_dir,
#     csv_path=csv_path,
#     gd_model=gd_model,
#     sam_predictor=sam_predictor,
#     device=device
# )