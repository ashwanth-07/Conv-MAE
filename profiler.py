#!/usr/bin/env python3
"""
Simple ConvMAE Timing Profiler
==============================

A comprehensive yet simple timing profiler for ConvMAE training.
Run this to see exactly what components are taking time during training.

Usage:
    python convmae_timer.py --backbone b2 --batch_size 8 --iterations 20
    
Output:
    - Console timing breakdown
    - JSON results file
    - Timing visualization plot
"""

import time
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, OrderedDict
import contextlib
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import your ConvMAE model
from convmae_models import ConvMAEPretrainer

class SimpleTimer:
    """Simple timing context manager with CUDA sync"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.timings = defaultdict(list)
    
    @contextlib.contextmanager
    def time(self, name: str):
        """Time a code block"""
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            
            try:
                yield
            finally:
                end.record()
                torch.cuda.synchronize()
                elapsed = start.elapsed_time(end)  # milliseconds
                self.timings[name].append(elapsed)
        else:
            start_time = time.time()
            try:
                yield
            finally:
                elapsed = (time.time() - start_time) * 1000  # convert to ms
                self.timings[name].append(elapsed)
    
    def get_stats(self):
        """Get timing statistics"""
        stats = {}
        for name, times in self.timings.items():
            if times:
                stats[name] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'total': np.sum(times),
                    'count': len(times)
                }
        return stats
    
    def reset(self):
        """Clear all timings"""
        self.timings.clear()

class ConvMAETimedModel(nn.Module):
    """ConvMAE model wrapper with detailed timing"""
    
    def __init__(self, model: ConvMAEPretrainer, timer: SimpleTimer):
        super().__init__()
        self.model = model
        self.timer = timer
    
    def forward(self, images, detailed=True):
        """Forward pass with detailed timing"""
        
        if not detailed:
            return self.model(images)
        
        with self.timer.time('0_total_forward'):
            # Time backbone execution
            with self.timer.time('1_backbone_total'):
                # Time individual backbone stages
                with self.timer.time('1a_mask_generation'):
                    mask = self.model.backbone.generate_mask(images) if self.model.backbone.mask_ratio else None
                
                # Execute backbone with stage-level timing
                backbone_output = {}
                backbone_output["input"] = images
                
                # Stage 0 (input stem)
                with self.timer.time('1b_backbone_stage0'):
                    x = self.model.backbone.input_stem(images, mask)
                backbone_output["stage0"] = x
                
                # Stages 1-4
                for stage_id, stage in enumerate(self.model.backbone.stages, 1):
                    with self.timer.time(f'1{chr(ord("b") + stage_id)}_backbone_stage{stage_id}'):
                        x = stage(x, mask)
                    backbone_output[f"stage{stage_id}"] = x
                
                backbone_output["stage_final"] = x
            
            # Time feature extraction
            with self.timer.time('2_feature_extraction'):
                multi_scale_features = [
                    backbone_output["stage0"],  # H/2
                    backbone_output["stage1"],  # H/4
                    backbone_output["stage2"],  # H/8
                    backbone_output["stage3"],  # H/16
                    backbone_output["stage4"],  # H/32
                ]
            
            # Time decoder components
            with self.timer.time('3_decoder_total'):
                # Time decoder forward pass
                with self.timer.time('3a_decoder_projection'):
                    # Project each stage to decoder resolution
                    projected_features = []
                    for feat, proj in zip(multi_scale_features, self.model.decoder.stage_projections):
                        proj_feat = proj(feat)
                        proj_feat = proj_feat.flatten(2).transpose(1, 2)
                        proj_feat = self.model.decoder.feature_norm(proj_feat)
                        projected_features.append(proj_feat)
                
                with self.timer.time('3b_decoder_fusion'):
                    # Fuse multi-scale features
                    fused = torch.cat(projected_features, dim=-1)
                    fused = self.model.decoder.multi_scale_fusion(fused)
                
                with self.timer.time('3c_decoder_masking'):
                    # Prepare decoder tokens with masking
                    _, _, H32, W32 = multi_scale_features[-1].shape
                    if mask.shape[-2:] != (H32, W32):
                        mask_resized = F.interpolate(mask.float(), size=(H32, W32), mode='nearest')
                    else:
                        mask_resized = mask.float()
                    mask_flat = mask_resized.flatten(2).transpose(1, 2).squeeze(-1)
                    
                    B = fused.shape[0]
                    decoder_tokens = torch.zeros(B, H32 * W32, self.model.decoder.decoder_dim, device=fused.device)
                    for b in range(B):
                        vis_idx = mask_flat[b].bool()
                        decoder_tokens[b, vis_idx] = fused[b, vis_idx]
                        masked_idx = ~vis_idx
                        if masked_idx.sum() > 0:
                            decoder_tokens[b, masked_idx] = self.model.decoder.mask_token.expand(masked_idx.sum(), self.model.decoder.decoder_dim)
                
                with self.timer.time('3d_decoder_pos_embed'):
                    # Add positional embedding
                    self.model.decoder._init_pos_embed(H32, W32)
                    decoder_tokens = self.model.decoder.pos_embed(decoder_tokens)
                
                with self.timer.time('3e_decoder_transformer'):
                    # Run through transformer blocks
                    for blk in self.model.decoder.decoder_blocks:
                        decoder_tokens = blk(decoder_tokens)
                
                with self.timer.time('3f_decoder_prediction'):
                    # Final prediction
                    decoder_tokens = self.model.decoder.decoder_norm(decoder_tokens)
                    pred = self.model.decoder.decoder_pred(decoder_tokens)
                    
                    # Reshape to image
                    pred = pred.view(B, H32, W32, self.model.decoder.patch_size, self.model.decoder.patch_size, self.model.decoder.in_channels)
                    pred = pred.permute(0, 5, 1, 3, 2, 4)
                    pred = pred.reshape(B, self.model.decoder.in_channels, H32 * self.model.decoder.patch_size, W32 * self.model.decoder.patch_size)
            
            # Time loss computation
            with self.timer.time('4_loss_computation'):
                loss = self.model.decoder.forward_loss(images, pred, mask)
        
        return pred, loss, mask

def create_synthetic_data(batch_size=8, image_size=224, device='cuda'):
    """Create synthetic data for timing"""
    return torch.randn(batch_size, 3, image_size, image_size, device=device)

def get_memory_usage():
    """Get current GPU memory usage in GB"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1e9,
            'cached': torch.cuda.memory_reserved() / 1e9,
            'max_allocated': torch.cuda.max_memory_allocated() / 1e9
        }
    return {'allocated': 0, 'cached': 0, 'max_allocated': 0}

def time_training_step(model, optimizer, images, timer):
    """Time a complete training step"""
    
    # Forward pass
    with timer.time('step_1_forward'):
        pred, loss, mask = model(images, detailed=True)
    
    # Backward pass
    with timer.time('step_2_backward'):
        loss.backward()
    
    # Optimizer step
    with timer.time('step_3_optimizer'):
        optimizer.step()
        optimizer.zero_grad()
    
    return loss.item()

def run_timing_analysis(
    backbone_name='b2',
    batch_size=8,
    image_size=224,
    iterations=20,
    warmup=5,
    device='cuda',
    test_memory=True
):
    """Run comprehensive timing analysis"""
    
    print(f"🔍 ConvMAE Timing Analysis")
    print(f"Model: {backbone_name}, Batch: {batch_size}, Device: {device}")
    print(f"Iterations: {iterations} (+ {warmup} warmup)")
    print("-" * 60)
    
    device = torch.device(device)
    timer = SimpleTimer(device)
    
    # Create model
    print("Creating model...")
    model = ConvMAEPretrainer(
        backbone_name=backbone_name,
        mask_ratio=0.75,
        decoder_dim=512,
        decoder_depth=8,
        patch_size=16
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Wrap model for timing
    timed_model = ConvMAETimedModel(model, timer)
    timed_model.train()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    
    # Warmup
    print(f"\nWarming up ({warmup} iterations)...")
    for i in range(warmup):
        images = create_synthetic_data(batch_size, image_size, device)
        loss = time_training_step(timed_model, optimizer, images, timer)
        if (i + 1) % max(1, warmup // 2) == 0:
            print(f"  Warmup {i+1}/{warmup}: Loss = {loss:.6f}")
    
    # Reset timings after warmup
    timer.reset()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    
    # Main timing loop
    print(f"\nTiming analysis ({iterations} iterations)...")
    losses = []
    memory_usage = []
    
    start_time = time.time()
    
    for i in range(iterations):
        images = create_synthetic_data(batch_size, image_size, device)
        
        loss = time_training_step(timed_model, optimizer, images, timer)
        losses.append(loss)
        
        # Track memory
        if test_memory and device.type == 'cuda':
            memory_usage.append(get_memory_usage())
        
        # Progress update
        if (i + 1) % max(1, iterations // 4) == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (iterations - i - 1)
            print(f"  Progress {i+1}/{iterations}: Loss = {loss:.6f}, "
                  f"ETA = {eta:.1f}s")
    
    total_time = time.time() - start_time
    
    # Get timing statistics
    stats = timer.get_stats()
    
    # Calculate derived metrics
    if stats:
        samples_per_second = (batch_size * iterations) / total_time
        ms_per_sample = (total_time * 1000) / (batch_size * iterations)
        
        print(f"\n📊 Performance Summary:")
        print(f"Total time: {total_time:.2f}s")
        print(f"Throughput: {samples_per_second:.1f} samples/sec")
        print(f"Time per sample: {ms_per_sample:.2f}ms")
        print(f"Average loss: {np.mean(losses):.6f}")
        
        if memory_usage:
            avg_memory = np.mean([m['allocated'] for m in memory_usage])
            max_memory = max([m['max_allocated'] for m in memory_usage])
            print(f"Average memory: {avg_memory:.2f}GB")
            print(f"Peak memory: {max_memory:.2f}GB")
    
    return stats, {
        'losses': losses,
        'memory_usage': memory_usage,
        'total_time': total_time,
        'throughput': samples_per_second if 'samples_per_second' in locals() else 0,
        'config': {
            'backbone': backbone_name,
            'batch_size': batch_size,
            'image_size': image_size,
            'iterations': iterations,
            'device': str(device)
        }
    }

def print_timing_breakdown(stats):
    """Print detailed timing breakdown"""
    
    if not stats:
        print("No timing data available!")
        return
    
    # Sort by mean time (descending)
    sorted_stats = OrderedDict(sorted(stats.items(), key=lambda x: x[1]['mean'], reverse=True))
    
    # Calculate percentages
    total_time = sum(s['mean'] for s in stats.values())
    
    print(f"\n📈 Detailed Timing Breakdown:")
    print(f"{'='*80}")
    print(f"{'Component':<30} {'Mean (ms)':<12} {'Std (ms)':<12} {'% Total':<10} {'Count':<8}")
    print(f"{'-'*80}")
    
    for name, stat in sorted_stats.items():
        percentage = (stat['mean'] / total_time) * 100 if total_time > 0 else 0
        print(f"{name:<30} {stat['mean']:<12.2f} {stat['std']:<12.2f} {percentage:<10.1f} {stat['count']:<8}")
    
    print(f"{'-'*80}")
    print(f"{'TOTAL':<30} {total_time:<12.2f} {'':<12} {'100.0':<10} {'':<8}")
    print(f"{'='*80}")
    
    # Highlight bottlenecks
    print(f"\n🔥 Top 3 Bottlenecks:")
    for i, (name, stat) in enumerate(list(sorted_stats.items())[:3], 1):
        percentage = (stat['mean'] / total_time) * 100
        print(f"{i}. {name}: {stat['mean']:.2f}ms ({percentage:.1f}%)")

def save_results(stats, metadata, output_file='timing_results.json'):
    """Save timing results to JSON"""
    
    results = {
        'timing_stats': stats,
        'metadata': metadata,
        'summary': {
            'total_components': len(stats),
            'total_time_ms': sum(s['mean'] for s in stats.values()),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")

def plot_timing_results(stats, output_file='timing_plot.png'):
    """Create timing visualization"""
    
    if not stats:
        return
    
    # Prepare data
    components = list(stats.keys())
    means = [stats[comp]['mean'] for comp in components]
    stds = [stats[comp]['std'] for comp in components]
    
    # Sort by mean time
    sorted_indices = np.argsort(means)[::-1]
    components = [components[i] for i in sorted_indices]
    means = [means[i] for i in sorted_indices]
    stds = [stds[i] for i in sorted_indices]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Bar plot
    bars = ax1.bar(range(len(components)), means, yerr=stds, capsize=3, alpha=0.7, color='steelblue')
    ax1.set_title('ConvMAE Component Timing', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_xlabel('Components', fontsize=12)
    ax1.set_xticks(range(len(components)))
    ax1.set_xticklabels(components, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(stds) * 0.1,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Pie chart for percentages > 2%
    total_time = sum(means)
    percentages = [(mean/total_time)*100 for mean in means]
    
    # Filter small components
    filtered_components = []
    filtered_percentages = []
    other_percentage = 0
    
    for comp, pct in zip(components, percentages):
        if pct >= 2.0:
            filtered_components.append(comp.replace('_', ' ').title())
            filtered_percentages.append(pct)
        else:
            other_percentage += pct
    
    if other_percentage > 0:
        filtered_components.append('Others')
        filtered_percentages.append(other_percentage)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(filtered_components)))
    wedges, texts, autotexts = ax2.pie(filtered_percentages, labels=filtered_components, 
                                      autopct='%1.1f%%', startangle=90, colors=colors)
    ax2.set_title('Time Distribution', fontsize=14, fontweight='bold')
    
    # Improve text readability
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plot saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='ConvMAE Timing Profiler')
    
    parser.add_argument('--backbone', default='b2', choices=['b0', 'b1', 'b2', 'b3'],
                        help='EfficientViT backbone model')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size for testing')
    parser.add_argument('--image_size', default=224, type=int,
                        help='Input image size')
    parser.add_argument('--iterations', default=20, type=int,
                        help='Number of timing iterations')
    parser.add_argument('--warmup', default=5, type=int,
                        help='Number of warmup iterations')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use for testing')
    parser.add_argument('--output_dir', default='./timing_results', type=str,
                        help='Output directory for results')
    parser.add_argument('--no_memory', action='store_true',
                        help='Skip memory profiling')
    parser.add_argument('--no_plot', action='store_true',
                        help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run timing analysis
    stats, metadata = run_timing_analysis(
        backbone_name=args.backbone,
        batch_size=args.batch_size,
        image_size=args.image_size,
        iterations=args.iterations,
        warmup=args.warmup,
        device=args.device,
        test_memory=not args.no_memory
    )
    
    # Print results
    print_timing_breakdown(stats)
    
    # Save results
    output_file = output_dir / f'timing_results_{args.backbone}_bs{args.batch_size}.json'
    save_results(stats, metadata, output_file)
    
    # Create plot
    if not args.no_plot:
        plot_file = output_dir / f'timing_plot_{args.backbone}_bs{args.batch_size}.png'
        plot_timing_results(stats, plot_file)
    
    print(f"\n✅ Analysis complete! Check {args.output_dir}/ for detailed results.")

if __name__ == '__main__':
    main()