"""
Standalone tool for comparing EfficientViT backbone implementations.
This helps verify that ConvMAE wrappers maintain full compatibility.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple


class BackboneComparator:
    """
    Comprehensive tool for comparing backbone architectures and ensuring compatibility.
    """
    
    def __init__(self, name: str = "Backbone Comparison"):
        self.name = name
        self.results = {}
        
    def compare_all(
        self, 
        backbone1, 
        backbone2, 
        name1: str = "Original", 
        name2: str = "Modified",
        test_input: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Run all comparison tests between two backbones.
        
        Args:
            backbone1: First backbone to compare
            backbone2: Second backbone to compare  
            name1: Name for first backbone
            name2: Name for second backbone
            test_input: Input tensor for testing (default: random 224x224)
            
        Returns:
            Dictionary with detailed comparison results
        """
        print("=" * 100)
        print(f"🔍 {self.name.upper()}")
        print("=" * 100)
        
        if test_input is None:
            test_input = torch.randn(1, 3, 224, 224)
            
        # Extract actual backbones if wrapped
        actual_backbone1 = self._extract_backbone(backbone1)
        actual_backbone2 = self._extract_backbone(backbone2)
        
        results = {
            'basic_info': self._compare_basic_info(actual_backbone1, actual_backbone2, name1, name2),
            'module_structure': self._compare_module_structure(actual_backbone1, actual_backbone2, name1, name2),
            'parameters': self._compare_parameters(actual_backbone1, actual_backbone2, name1, name2),
            'forward_pass': self._compare_forward_pass(actual_backbone1, actual_backbone2, test_input, name1, name2),
            'state_dict': self._compare_state_dicts(actual_backbone1, actual_backbone2, name1, name2)
        }
        
        # Generate summary
        results['summary'] = self._generate_summary(results)
        
        self.results = results
        return results
    
    def _extract_backbone(self, backbone):
        """Extract actual backbone from potential wrapper."""
        if hasattr(backbone, 'backbone'):
            return backbone.backbone
        return backbone
    
    def _compare_basic_info(self, backbone1, backbone2, name1: str, name2: str) -> Dict[str, Any]:
        """Compare basic information about the backbones."""
        print(f"\n📋 BASIC INFORMATION")
        print("-" * 50)
        
        info1 = {
            'type': type(backbone1).__name__,
            'module': type(backbone1).__module__,
            'width_list': getattr(backbone1, 'width_list', None),
            'total_params': sum(p.numel() for p in backbone1.parameters()),
            'trainable_params': sum(p.numel() for p in backbone1.parameters() if p.requires_grad)
        }
        
        info2 = {
            'type': type(backbone2).__name__,
            'module': type(backbone2).__module__,
            'width_list': getattr(backbone2, 'width_list', None),
            'total_params': sum(p.numel() for p in backbone2.parameters()),
            'trainable_params': sum(p.numel() for p in backbone2.parameters() if p.requires_grad)
        }
        
        print(f"{name1:15} | {name2:15} | Match")
        print("-" * 50)
        
        matches = {}
        for key in info1.keys():
            val1, val2 = info1[key], info2[key]
            match = val1 == val2
            matches[key] = match
            
            if key in ['total_params', 'trainable_params']:
                print(f"{key:15} | {val1:>13,} | {val2:>13,} | {'✓' if match else '✗'}")
            else:
                val1_str = str(val1)[:12] + "..." if len(str(val1)) > 15 else str(val1)
                val2_str = str(val2)[:12] + "..." if len(str(val2)) > 15 else str(val2)
                print(f"{key:15} | {val1_str:>13} | {val2_str:>13} | {'✓' if match else '✗'}")
        
        return {
            name1: info1,
            name2: info2,
            'matches': matches,
            'all_match': all(matches.values())
        }
    
    def _compare_module_structure(self, backbone1, backbone2, name1: str, name2: str) -> Dict[str, Any]:
        """Compare module structure between backbones."""
        print(f"\n🏗️  MODULE STRUCTURE")
        print("-" * 50)
        
        modules1 = dict(backbone1.named_modules())
        modules2 = dict(backbone2.named_modules())
        
        names1 = set(modules1.keys())
        names2 = set(modules2.keys())
        
        common = names1 & names2
        only_1 = names1 - names2
        only_2 = names2 - names1
        
        print(f"Total modules:     {name1}: {len(modules1)}, {name2}: {len(modules2)}")
        print(f"Common modules:    {len(common)}")
        print(f"Only in {name1}:    {len(only_1)}")
        print(f"Only in {name2}:    {len(only_2)}")
        
        # Check type matches for common modules
        type_matches = 0
        type_mismatches = []
        
        for name in common:
            if type(modules1[name]) == type(modules2[name]):
                type_matches += 1
            else:
                type_mismatches.append({
                    'name': name,
                    'type1': type(modules1[name]).__name__,
                    'type2': type(modules2[name]).__name__
                })
        
        print(f"Type matches:      {type_matches}/{len(common)}")
        
        # Print detailed differences if any
        if only_1:
            print(f"\n❌ Only in {name1}:")
            for name in sorted(list(only_1)[:10]):  # Limit to first 10
                print(f"  - {name} ({type(modules1[name]).__name__})")
            if len(only_1) > 10:
                print(f"  ... and {len(only_1) - 10} more")
        
        if only_2:
            print(f"\n➕ Only in {name2}:")
            for name in sorted(list(only_2)[:10]):  # Limit to first 10
                print(f"  + {name} ({type(modules2[name]).__name__})")
            if len(only_2) > 10:
                print(f"  ... and {len(only_2) - 10} more")
        
        if type_mismatches:
            print(f"\n🔄 Type mismatches:")
            for mismatch in type_mismatches[:5]:  # Limit to first 5
                print(f"  {mismatch['name']}: {mismatch['type1']} vs {mismatch['type2']}")
            if len(type_mismatches) > 5:
                print(f"  ... and {len(type_mismatches) - 5} more")
        
        return {
            'modules_count': {name1: len(modules1), name2: len(modules2)},
            'common_modules': len(common),
            'only_1': len(only_1),
            'only_2': len(only_2),
            'type_matches': type_matches,
            'type_mismatches': len(type_mismatches),
            'structure_identical': len(only_1) == 0 and len(only_2) == 0 and len(type_mismatches) == 0
        }
    
    def _compare_parameters(self, backbone1, backbone2, name1: str, name2: str) -> Dict[str, Any]:
        """Compare parameters between backbones."""
        print(f"\n🔢 PARAMETERS")
        print("-" * 50)
        
        params1 = dict(backbone1.named_parameters())
        params2 = dict(backbone2.named_parameters())
        
        names1 = set(params1.keys())
        names2 = set(params2.keys())
        
        common = names1 & names2
        only_1 = names1 - names2
        only_2 = names2 - names1
        
        print(f"Total parameters:  {name1}: {len(params1)}, {name2}: {len(params2)}")
        print(f"Common parameters: {len(common)}")
        print(f"Only in {name1}:     {len(only_1)}")
        print(f"Only in {name2}:     {len(only_2)}")
        
        # Check shape matches for common parameters
        shape_matches = 0
        shape_mismatches = []
        value_matches = 0
        
        for name in common:
            param1, param2 = params1[name], params2[name]
            
            # Check shapes
            if param1.shape == param2.shape:
                shape_matches += 1
                
                # Check values (only if shapes match)
                if torch.allclose(param1, param2, atol=1e-6):
                    value_matches += 1
            else:
                shape_mismatches.append({
                    'name': name,
                    'shape1': param1.shape,
                    'shape2': param2.shape
                })
        
        print(f"Shape matches:     {shape_matches}/{len(common)}")
        print(f"Value matches:     {value_matches}/{len(common)}")
        
        # Show shape mismatches
        if shape_mismatches:
            print(f"\n📐 Shape mismatches:")
            for mismatch in shape_mismatches[:5]:
                print(f"  {mismatch['name']}: {mismatch['shape1']} vs {mismatch['shape2']}")
            if len(shape_mismatches) > 5:
                print(f"  ... and {len(shape_mismatches) - 5} more")
        
        # Show missing parameters
        if only_1:
            print(f"\n❌ Missing in {name2}:")
            for name in sorted(list(only_1)[:5]):
                print(f"  - {name} {params1[name].shape}")
            if len(only_1) > 5:
                print(f"  ... and {len(only_1) - 5} more")
        
        if only_2:
            print(f"\n➕ Extra in {name2}:")
            for name in sorted(list(only_2)[:5]):
                print(f"  + {name} {params2[name].shape}")
            if len(only_2) > 5:
                print(f"  ... and {len(only_2) - 5} more")
        
        return {
            'param_count': {name1: len(params1), name2: len(params2)},
            'common_params': len(common),
            'only_1': len(only_1),
            'only_2': len(only_2),
            'shape_matches': shape_matches,
            'value_matches': value_matches,
            'shape_mismatches': len(shape_mismatches),
            'parameters_identical': len(only_1) == 0 and len(only_2) == 0 and len(shape_mismatches) == 0
        }
    
    def _compare_forward_pass(self, backbone1, backbone2, test_input: torch.Tensor, name1: str, name2: str) -> Dict[str, Any]:
        """Compare forward pass outputs."""
        print(f"\n🚀 FORWARD PASS")
        print("-" * 50)
        
        try:
            backbone1.eval()
            backbone2.eval()
            
            with torch.no_grad():
                output1 = backbone1(test_input)
                output2 = backbone2(test_input)
            
            # Handle different output types
            if isinstance(output1, dict) and isinstance(output2, dict):
                return self._compare_dict_outputs(output1, output2, name1, name2)
            elif torch.is_tensor(output1) and torch.is_tensor(output2):
                return self._compare_tensor_outputs(output1, output2, name1, name2)
            else:
                print(f"❌ Output type mismatch: {type(output1)} vs {type(output2)}")
                return {
                    'success': False,
                    'error': 'Output type mismatch',
                    'compatible': False
                }
                
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'compatible': False
            }
    
    def _compare_dict_outputs(self, output1: dict, output2: dict, name1: str, name2: str) -> Dict[str, Any]:
        """Compare dictionary outputs."""
        keys1, keys2 = set(output1.keys()), set(output2.keys())
        common_keys = keys1 & keys2
        
        print(f"Output keys:       {name1}: {sorted(keys1)}")
        print(f"                   {name2}: {sorted(keys2)}")
        print(f"Common keys:       {len(common_keys)}")
        
        key_match = keys1 == keys2
        shape_matches = 0
        value_matches = 0
        
        if key_match:
            print(f"✓ Output keys match")
            
            for key in common_keys:
                if torch.is_tensor(output1[key]) and torch.is_tensor(output2[key]):
                    shape_match = output1[key].shape == output2[key].shape
                    if shape_match:
                        shape_matches += 1
                        value_match = torch.allclose(output1[key], output2[key], atol=1e-6)
                        if value_match:
                            value_matches += 1
                        print(f"  {key:15}: {output1[key].shape} | values_match: {value_match}")
                    else:
                        print(f"  {key:15}: shape mismatch {output1[key].shape} vs {output2[key].shape}")
                        
            compatible = shape_matches == len(common_keys) and value_matches == len(common_keys)
        else:
            print(f"✗ Output keys don't match")
            compatible = False
        
        return {
            'success': True,
            'key_match': key_match,
            'shape_matches': shape_matches,
            'value_matches': value_matches,
            'total_tensor_outputs': len(common_keys),
            'compatible': compatible
        }
    
    def _compare_tensor_outputs(self, output1: torch.Tensor, output2: torch.Tensor, name1: str, name2: str) -> Dict[str, Any]:
        """Compare tensor outputs."""
        shape_match = output1.shape == output2.shape
        print(f"Output shapes:     {name1}: {output1.shape}")
        print(f"                   {name2}: {output2.shape}")
        print(f"Shape match:       {shape_match}")
        
        if shape_match:
            value_match = torch.allclose(output1, output2, atol=1e-6)
            print(f"Value match:       {value_match}")
            compatible = value_match
        else:
            compatible = False
            value_match = False
        
        return {
            'success': True,
            'shape_match': shape_match,
            'value_match': value_match,
            'compatible': compatible
        }
    
    def _compare_state_dicts(self, backbone1, backbone2, name1: str, name2: str) -> Dict[str, Any]:
        """Compare state dictionaries."""
        print(f"\n💾 STATE DICT COMPATIBILITY")
        print("-" * 50)
        
        try:
            state_dict1 = backbone1.state_dict()
            state_dict2 = backbone2.state_dict()
            
            keys1, keys2 = set(state_dict1.keys()), set(state_dict2.keys())
            common_keys = keys1 & keys2
            
            print(f"State dict keys:   {name1}: {len(keys1)}, {name2}: {len(keys2)}")
            print(f"Common keys:       {len(common_keys)}")
            
            # Test loading state dict
            from efficientvit.models.backbone import efficientvit_backbone_b2
            test_backbone = efficientvit_backbone_b2()
            
            missing, unexpected = test_backbone.load_state_dict(state_dict2, strict=False)
            
            print(f"Load test:         missing: {len(missing)}, unexpected: {len(unexpected)}")
            
            loadable = len(missing) == 0
            
            return {
                'success': True,
                'keys_match': keys1 == keys2,
                'common_keys': len(common_keys),
                'missing_keys': len(keys1 - keys2),
                'extra_keys': len(keys2 - keys1),
                'loadable': loadable,
                'load_missing': len(missing),
                'load_unexpected': len(unexpected)
            }
            
        except Exception as e:
            print(f"❌ State dict comparison failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall compatibility summary."""
        print(f"\n🎯 COMPATIBILITY SUMMARY")
        print("=" * 50)
        
        checks = []
        
        # Basic info checks
        if results['basic_info']['all_match']:
            checks.append(('Basic Info', True, 'All basic properties match'))
            print("✅ Basic properties match")
        else:
            checks.append(('Basic Info', False, 'Basic properties differ'))
            print("❌ Basic properties differ")
        
        # Structure checks
        if results['module_structure']['structure_identical']:
            checks.append(('Module Structure', True, 'Module structures identical'))
            print("✅ Module structures identical")
        else:
            checks.append(('Module Structure', False, 'Module structures differ'))
            print("❌ Module structures differ")
        
        # Parameter checks
        if results['parameters']['parameters_identical']:
            checks.append(('Parameters', True, 'Parameters identical'))
            print("✅ Parameters identical")
        else:
            checks.append(('Parameters', False, 'Parameters differ'))
            print("❌ Parameters differ")
        
        # Forward pass checks
        if results['forward_pass']['success'] and results['forward_pass']['compatible']:
            checks.append(('Forward Pass', True, 'Forward passes compatible'))
            print("✅ Forward passes compatible")
        else:
            checks.append(('Forward Pass', False, 'Forward passes incompatible'))
            print("❌ Forward passes incompatible")
        
        # State dict checks
        if results['state_dict']['success'] and results['state_dict']['loadable']:
            checks.append(('State Dict', True, 'State dict loadable'))
            print("✅ State dict loadable")
        else:
            checks.append(('State Dict', False, 'State dict issues'))
            print("❌ State dict issues")
        
        passed_checks = sum(1 for _, passed, _ in checks if passed)
        total_checks = len(checks)
        compatibility_score = (passed_checks / total_checks) * 100
        
        print(f"\nCompatibility Score: {passed_checks}/{total_checks} ({compatibility_score:.0f}%)")
        
        if compatibility_score >= 90:
            status = "EXCELLENT"
            recommendation = "Safe for all downstream use cases"
            print("🟢 EXCELLENT compatibility")
        elif compatibility_score >= 70:
            status = "GOOD"
            recommendation = "Generally safe, monitor for edge cases"
            print("🟡 GOOD compatibility")
        elif compatibility_score >= 50:
            status = "POOR"
            recommendation = "Use with caution, significant issues present"
            print("🟠 POOR compatibility")
        else:
            status = "CRITICAL"
            recommendation = "Not recommended for use, major incompatibilities"
            print("🔴 CRITICAL compatibility issues")
        
        return {
            'checks': checks,
            'passed_checks': passed_checks,
            'total_checks': total_checks,
            'compatibility_score': compatibility_score,
            'status': status,
            'recommendation': recommendation
        }
    
    def save_report(self, filepath: str):
        """Save detailed comparison report."""
        if not self.results:
            print("No comparison results to save. Run compare_all() first.")
            return
            
        report = {
            'timestamp': torch.datetime.now().isoformat(),
            'comparison_name': self.name,
            'results': self.results
        }
        
        torch.save(report, filepath)
        print(f"Comparison report saved to {filepath}")


def main():
    """Main function to run backbone comparison."""
    print("🔍 EfficientViT Backbone Comparison Tool")
    print("=" * 100)
    
    # Import required modules
    try:
        from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b2
        from backbone import RobustConvMAEWrapper
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure EfficientViT and ConvMAE implementations are available.")
        return
    
    # Create backbones
    print("🏗️  Creating backbones...")
    original_backbone = efficientvit_backbone_b2()
    convmae_wrapper = RobustConvMAEWrapper(original_backbone, mask_ratio=0.75)
    
    # Run comparison
    comparator = BackboneComparator("EfficientViT vs ConvMAE Wrapper")
    results = comparator.compare_all(
        original_backbone, 
        convmae_wrapper,
        "EfficientViT-B2",
        "ConvMAE Wrapper"
    )
    
    # Save report
    comparator.save_report('backbone_comparison_report.pt')
    
    print(f"\n✅ Comparison completed!")
    print(f"Compatibility Score: {results['summary']['compatibility_score']:.0f}%")
    print(f"Status: {results['summary']['status']}")
    print(f"Recommendation: {results['summary']['recommendation']}")


if __name__ == "__main__":
    main()