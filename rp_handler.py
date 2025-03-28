import runpod
import time

def handler(event):
    input = event['input']
    instruction = input.get('instruction')
    seconds = input.get('seconds', 0)

    # Placeholder for a task; replace with image or text generation logic as needed
    time.sleep(seconds)
    result = instruction.replace(instruction.split()[0], 'created', 1)

    return result

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})