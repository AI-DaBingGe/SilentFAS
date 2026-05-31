import os
import torch
from src.utility import parse_model_name, get_kernel
from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE
}

def export_to_onnx(model_dir):
    for model_name in os.listdir(model_dir):
        if not model_name.endswith('.pth'):
            continue
            
        print(f"Converting {model_name} to ONNX...")
        model_path = os.path.join(model_dir, model_name)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        kernel_size = get_kernel(h_input, w_input)
        model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size)
        
        # load model weight
        state_dict = torch.load(model_path, map_location='cpu')
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
            
        model.eval()
        
        # Dummy input for ONNX export (Batch Size = 1, Channels = 3, H, W)
        dummy_input = torch.randn(1, 3, h_input, w_input, device='cpu')
        
        onnx_filename = model_name.replace('.pth', '.onnx')
        onnx_path = os.path.join(model_dir, onnx_filename)
        
        import sys
        class DummyFile(object):
            def write(self, x): pass
            def flush(self): pass
            
        old_stdout = sys.stdout
        sys.stdout = DummyFile()
        try:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output']
            )
        finally:
            sys.stdout = old_stdout
        print(f"Successfully exported to {onnx_path}")

if __name__ == '__main__':
    export_to_onnx("./resources/anti_spoof_models")
