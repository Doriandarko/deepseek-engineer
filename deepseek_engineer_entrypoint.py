import runpy
import os

def main():
    # Run the actual script file
    script_path = os.path.join(os.path.dirname(__file__), 'deepseek-eng.py')
    runpy.run_path(script_path, run_name='__main__')

if __name__ == '__main__':
    main()
