import onnx
import onnxruntime as ort
import os
from onnx import checker, helper

def validate_onnx_model(model_path):
    """
    Validate an ONNX model by checking its structure and loading it into ONNX Runtime.
    
    Args:
        model_path (str): Path to the ONNX model file.
    
    Returns:
        bool: True if the model is valid and can be loaded, False otherwise.
    """
    print(f"Validating ONNX model: {model_path}")

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False

    try:
        # Load the model
        print("Loading ONNX model...")
        model = onnx.load(model_path)
        print("Model loaded successfully.")

        # Check model syntax and structure
        print("Checking model integrity...")
        checker.check_model(model)
        print("Model structure is valid.")

        # Get and print opset version
        opset_version = model.opset_import[0].version
        print(f"Model opset version: {opset_version}")

        # Verify model inputs and outputs
        print("Model inputs:")
        for input in model.graph.input:
            print(f" - {input.name}: {helper.printable_type(input.type)}")
        print("Model outputs:")
        for output in model.graph.output:
            print(f" - {output.name}: {helper.printable_type(output.type)}")

        # Test loading the model in ONNX Runtime
        print("Loading model into ONNX Runtime...")
        session = ort.InferenceSession(model_path)
        print("Model successfully loaded into ONNX Runtime.")

        # Print ONNX Runtime version
        print(f"ONNX Runtime version: {ort.__version__}")

        return True

    except Exception as e:
        print(f"Validation failed: {str(e)}")
        return False

def main():
    # Path to the ONNX model
    model_path = "app/model/remx_model_1.0.0.onnx"

    # Validate the model
    if validate_onnx_model(model_path):
        print("Model validation passed!")
    else:
        print("Model validation failed.")

if __name__ == "__main__":
    main()