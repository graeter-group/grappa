
import sys
import os

class no_print:
    """
    Context manager to suppress stdout.
    """
    def __enter__(self):
        # Save the original stdout
        self.original_stdout = sys.stdout 
        # Set stdout to null device
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Reset stdout to original
        sys.stdout = self.original_stdout
