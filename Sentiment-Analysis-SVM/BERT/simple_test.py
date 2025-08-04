import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def test_trained_bert(force_cpu=False):
    print("Simple BERT Test")
    print("=" * 40)
    
    if force_cpu:
        print("Forcing CPU mode (simulating cloud deployment)")
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
    
    # Check if model directory exists
    model_dir = "./bert_sentiment_model"
    if not os.path.exists(model_dir):
        print(f"Model directory not found: {model_dir}")
        return
    
    print(f"Model directory found: {model_dir}")
    
    # List contents
    print("Model directory contents:")
    for item in os.listdir(model_dir):
        print(f"   - {item}")
    
    # Try to load the model
    try:
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        print("Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            num_labels=3,
            ignore_mismatched_sizes=True
        )
        
        # Force CPU if requested
        if force_cpu:
            model = model.to("cpu")
            print("Model moved to CPU")
        
        print("Model loaded successfully!")
        
        # Create pipeline with device specification
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
        
        # Test with a simple comment
        test_comment = "I love this product!"
        print(f"\nTesting with: '{test_comment}'")
        
        result = classifier(test_comment)[0]
        print(f"Result: {result['label']} (Confidence: {result['score']:.2%})")
        
        # Interactive testing
        print("\nInteractive Testing (type 'quit' to exit):")
        while True:
            comment = input("\nEnter comment: ").strip()
            if comment.lower() in ['quit', 'exit', 'q']:
                break
            
            if comment:
                try:
                    result = classifier(comment)[0]
                    print(f"Result: {result['label']} (Confidence: {result['score']:.2%})")
                except Exception as e:
                    print(f"Error: {e}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative approach...")
        
        # Alternative: try loading from checkpoint directly
        try:
            checkpoint_path = "./bert_sentiment_model/checkpoint-10138"
            print(f"Trying checkpoint: {checkpoint_path}")
            
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint_path,
                num_labels=3,
                ignore_mismatched_sizes=True
            )
            
            # Force CPU if requested
            if force_cpu:
                model = model.to("cpu")
                print("Model moved to CPU")
            
            print("Model loaded from checkpoint!")
            
            classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
            
            # Test
            test_comment = "I love this product!"
            result = classifier(test_comment)[0]
            print(f"Test result: {result['label']} (Confidence: {result['score']:.2%})")
            
            # Interactive testing
            print("\nInteractive Testing (type 'quit' to exit):")
            print("Try comments like:")
            print("   - 'This is amazing!' (positive)")
            print("   - 'I hate this product' (negative)")
            print("   - 'It's okay, nothing special' (neutral)")
            print("-" * 50)
            
            while True:
                comment = input("\nEnter your comment: ").strip()
                if comment.lower() in ['quit', 'exit', 'q']:
                    print("Thanks for testing!")
                    break
                
                if not comment:
                    print("Please enter a comment!")
                    continue
                
                try:
                    result = classifier(comment)[0]
                    
                    # Convert label to readable format
                    label_map = {
                        'LABEL_0': 'NEGATIVE',
                        'LABEL_1': 'NEUTRAL', 
                        'LABEL_2': 'POSITIVE'
                    }
                    
                    sentiment = label_map.get(result['label'], result['label'])
                    confidence = result['score']
                    
                    print(f"Sentiment: {sentiment}")
                    print(f"Confidence: {confidence:.2%}")
                    print(f"Raw Label: {result['label']}")
                    
                except Exception as e:
                    print(f"Error analyzing comment: {e}")
            
        except Exception as e2:
            print(f"Checkpoint loading failed: {e2}")

if __name__ == "__main__":
    import sys
    
    # Check if user wants CPU testing
    if len(sys.argv) > 1 and sys.argv[1].lower() in ['cpu', '--cpu', '-c']:
        test_trained_bert(force_cpu=True)
    else:
        print("Usage:")
        print("  python simple_test.py          # Use GPU if available, else CPU")
        print("  python simple_test.py cpu      # Force CPU mode (cloud simulation)")
        print()
        test_trained_bert(force_cpu=False) 