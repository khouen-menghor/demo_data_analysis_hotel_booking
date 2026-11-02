import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cleaning import clean_and_save

if __name__ == "__main__":
    clean_and_save()
