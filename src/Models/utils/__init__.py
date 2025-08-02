from .model_factory import get_model

# Make ConcreteEncoder import conditional to avoid PyTorch conflicts
def get_concrete_encoder():
    """Lazily import ConcreteEncoder to avoid PyTorch import conflicts."""
    try:
        from .ConcreteEncoderLayer import ConcreteEncoder
        return ConcreteEncoder
    except ImportError as e:
        raise ImportError(f"ConcreteEncoder requires PyTorch: {e}")

# Export ConcreteEncoder through lazy loading
import sys
class ConcreteEncoderProxy:
    def __getattr__(self, name):
        ConcreteEncoder = get_concrete_encoder()
        return getattr(ConcreteEncoder, name)
    
    def __call__(self, *args, **kwargs):
        ConcreteEncoder = get_concrete_encoder()
        return ConcreteEncoder(*args, **kwargs)

ConcreteEncoder = ConcreteEncoderProxy()

__all__ = ['get_model', 'ConcreteEncoder'] 