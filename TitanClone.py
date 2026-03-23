from flask import Flask, jsonify, request, Response
import webview
import threading
from backend_logic import TitanCloneBackend

app = Flask(__name__)
backend = TitanCloneBackend()

def start_server():
    app.run(port=5000)


@app.route('/api/load_inference_model', methods=['POST'])
def load_inference_model_api():
    data = request.json
    response = backend.load_inference_model(
        model_path=data.get('model_path'),
        config_path=data.get('config_path'),
        device=data.get('device'),
        cluster_model_path=data.get('cluster_model_path', ''),
        enhance=data.get('enhance', False),
        diffusion_model_path=data.get('diffusion_model_path', ''),
        diffusion_config_path=data.get('diffusion_config_path', ''),
        shallow_diffusion=data.get('shallow_diffusion', False),
        only_diffusion=data.get('only_diffusion', False),
        use_spk_mix=data.get('use_spk_mix', False),
        feature_retrieval=data.get('feature_retrieval', False)
    )
    return jsonify(response)

@app.route('/api/run_inference', methods=['POST'])
def run_inference_api():
    data = request.json
    response = backend.run_inference(
        raw_audio_path=data.get('raw_audio_path'),
        spk=data.get('spk'),
        tran=data.get('tran'),
        slice_db=data.get('slice_db'),
        cluster_infer_ratio=data.get('cluster_infer_ratio'),
        auto_predict_f0=data.get('auto_predict_f0'),
        noice_scale=data.get('noice_scale'),
        pad_seconds=data.get('pad_seconds'),
        clip=data.get('clip'),
        lg=data.get('lg'),
        lgr=data.get('lgr'),
        f0p=data.get('f0p'),
        enhancer_adaptive_key=data.get('enhancer_adaptive_key'),
        cr_threshold=data.get('cr_threshold'),
        k_step=data.get('k_step'),
        use_spk_mix=data.get('use_spk_mix'),
        second_encoding=data.get('second_encoding'),
        loudness_envelope_adjustment=data.get('loudness_envelope_adjustment'),
        wav_format=data.get('wav_format', 'flac')
    )
    return jsonify(response)

@app.route('/api/preprocess_dataset', methods=['POST'])
def preprocess_dataset_api():
    data = request.json
    # For streaming output, we will use Response and a generator function
    def generate():
        for output_line in backend.preprocess_dataset(
            dataset_path=data.get('dataset_path'),
            speech_encoder=data.get('speech_encoder'),
            vol_aug=data.get('vol_aug', False),
            num_processes=data.get('num_processes', 1)
        ):
            yield f"data: {output_line}\n\n"
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/generate_hubert_f0', methods=['POST'])
def generate_hubert_f0_api():
    data = request.json
    def generate():
        for output_line in backend.generate_hubert_f0(
            f0_predictor=data.get('f0_predictor'),
            use_diff=data.get('use_diff', False),
            num_processes=data.get('num_processes', 1)
        ):
            yield f"data: {output_line}\n\n"
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/train_model', methods=['POST'])
def train_model_api():
    data = request.json
    def generate():
        for output_line in backend.train_model(
            config_path=data.get('config_path'),
            model_name=data.get('model_name')
        ):
            yield f"data: {output_line}\n\n"
    return Response(generate(), mimetype='text/event-stream')

@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TitanClone - Voice Conversion</title>
    <style>
        body { font-family: sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }
        .tabs { display: flex; margin-bottom: 20px; }
        .tab-button { padding: 10px 20px; cursor: pointer; border: 1px solid #ccc; border-bottom: none; background-color: #e0e0e0; }
        .tab-button.active { background-color: #fff; border-color: #fff; }
        .tab-content { border: 1px solid #ccc; padding: 20px; background-color: #fff; }
        .hidden { display: none; }
        .card { background-color: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .card h3 { margin-top: 0; color: #333; }
        .card div { margin-bottom: 15px; }
        .card label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
        .card input[type="text"], .card input[type="number"] { width: calc(100% - 22px); padding: 10px; border: 1px solid #ccc; border-radius: 4px; }
        .card button { background-color: #007bff; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        .card button:hover { background-color: #0056b3; }
        .output-area { background-color: #e9e9e9; border: 1px solid #ccc; border-radius: 4px; padding: 10px; min-height: 100px; max-height: 300px; overflow-y: auto; margin-top: 15px; font-family: monospace; white-space: pre-wrap; word-break: break-all; }
        .status-message { margin-top: 10px; font-weight: bold; }
    </style>
</head>
<body>
    <h1>🎤 TitanClone</h1>
    <p><strong>Your AI Singing Voice Clone</strong></p>
    <hr>

    <div class="tabs">
        <div class="tab-button active" onclick="showTab('inference', event)">Inference</div>
        <div class="tab-button" onclick="showTab('training', event)">Training</div>
    </div>

    <div id="inference-tab" class="tab-content">
        <h2>Inference Settings</h2>
        <div class="card">
            <h3>Load Model</h3>
            <div>
                <label for="modelPath">Model Path:</label>
                <input type="text" id="modelPath" value="logs/44k/G_37600.pth">
            </div>
            <div>
                <label for="configPath">Config Path:</label>
                <input type="text" id="configPath" value="logs/44k/config.json">
            </div>
            <button onclick="loadInferenceModel()">Load Inference Model</button>
            <p id="modelStatus"></p>
        </div>

        <div class="card">
            <h3>Run Inference</h3>
            <div>
                <label for="rawAudioPath">Raw Audio Path:</label>
                <input type="text" id="rawAudioPath" value="raw/your_audio.wav">
            </div>
            <div>
                <label for="speaker">Target Speaker (spk):</label>
                <input type="text" id="speaker" value="buyizi">
            </div>
            <div>
                <label for="transpose">Transpose (tran, semitones):</label>
                <input type="number" id="transpose" value="0">
            </div>
            <button onclick="runInference()">Run Inference</button>
            <p id="inferenceStatus" class="status-message"></p>
            <h4>Output:</h4>
            <div id="inferenceOutput" class="output-area"></div>
        </div>
    </div>

    <div id="training-tab" class="tab-content hidden">
        <h2>Training Settings</h2>

        <div class="card">
            <h3>1. Preprocess Dataset</h3>
            <div>
                <label for="datasetPath">Dataset Path:</label>
                <input type="text" id="datasetPath" value="./dataset/44k">
            </div>
            <div>
                <label for="speechEncoder">Speech Encoder:</label>
                <select id="speechEncoder">
                    <option value="vec768l12">vec768l12</option>
                    <option value="vec768l9">vec768l9</option>
                </select>
            </div>
            <div>
                <input type="checkbox" id="volAug"> <label for="volAug">Volume Augmentation</label>
            </div>
            <div>
                <label for="preprocessNumProcesses">Number of Processes:</label>
                <input type="number" id="preprocessNumProcesses" value="1" min="1">
            </div>
            <button onclick="preprocessDataset()">Start Preprocessing</button>
        </div>

        <div class="card">
            <h3>2. Generate Hubert & F0</h3>
            <div>
                <label for="f0Predictor">F0 Predictor:</label>
                <select id="f0Predictor">
                    <option value="pm">pm</option>
                    <option value="dio">dio</option>
                    <option value="harvest">harvest</option>
                    <option value="crepe">crepe</option>
                    <option value="rmvpe">rmvpe</option>
                    <option value="fcpe">fcpe</option>
                </select>
            </div>
            <div>
                <input type="checkbox" id="useDiff"> <label for="useDiff">Use Diffusion (for shallow diffusion)</label>
            </div>
            <div>
                <label for="hubertNumProcesses">Number of Processes:</label>
                <input type="number" id="hubertNumProcesses" value="1" min="1">
            </div>
            <button onclick="generateHubertF0()">Start F0 Generation</button>
        </div>

        <div class="card">
            <h3>3. Train Model</h3>
            <div>
                <label for="trainConfigPath">Config Path:</label>
                <input type="text" id="trainConfigPath" value="configs/config.json">
            </div>
            <div>
                <label for="modelName">Model Name (e.g., 44k):</label>
                <input type="text" id="modelName" value="44k">
            </div>
            <button onclick="trainModel()">Start Training</button>
        </div>

        <h4>Training Output:</h4>
        <div id="trainingOutput" class="output-area"></div>
    </div>

    <script>
        function showTab(tabId, event) {
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.add('hidden'));
            document.getElementById(tabId + '-tab').classList.remove('hidden');

            document.querySelectorAll('.tab-button').forEach(button => button.classList.remove('active'));
            event.target.classList.add('active');
        }

        async function loadInferenceModel() {
            const modelPath = document.getElementById('modelPath').value;
            const configPath = document.getElementById('configPath').value;
            const modelStatus = document.getElementById('modelStatus');
            modelStatus.textContent = 'Loading model...';
            modelStatus.style.color = 'orange';

            try {
                const response = await fetch('/api/load_inference_model', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ model_path: modelPath, config_path: configPath })
                });
                const result = await response.json();

                if (result.status === 'success') {
                    modelStatus.textContent = result.message;
                    modelStatus.style.color = 'green';
                } else {
                    modelStatus.textContent = result.message;
                    modelStatus.style.color = 'red';
                }
            } catch (error) {
                modelStatus.textContent = `Error: ${error.message}`;
                modelStatus.style.color = 'red';
            }
        }

        async function runInference() {
            const rawAudioPath = document.getElementById('rawAudioPath').value;
            const speaker = document.getElementById('speaker').value;
            const transpose = parseInt(document.getElementById('transpose').value);
            const inferenceStatus = document.getElementById('inferenceStatus');
            const inferenceOutput = document.getElementById('inferenceOutput');
            inferenceStatus.textContent = 'Running inference...';
            inferenceStatus.style.color = 'orange';
            inferenceOutput.textContent = '';

            try {
                const response = await fetch('/api/run_inference', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        raw_audio_path: rawAudioPath,
                        spk: speaker,
                        tran: transpose,
                        slice_db: -40, // Default value for now
                        cluster_infer_ratio: 0, // Default
                        auto_predict_f0: false, // Default
                        noice_scale: 0.4, // Default
                        pad_seconds: 0.5, // Default
                        clip: 0, // Default
                        lg: 0, // Default
                        lgr: 0.75, // Default
                        f0p: 'pm', // Default
                        enhancer_adaptive_key: 0, // Default
                        cr_threshold: 0.05, // Default
                        k_step: 100, // Default
                        use_spk_mix: false, // Default
                        second_encoding: false, // Default
                        loudness_envelope_adjustment: 1, // Default
                        wav_format: 'flac' // Default
                    })
                });
                const result = await response.json();

                if (result.status === 'success') {
                    inferenceStatus.textContent = result.message + ` Output: ${result.output_path}`;
                    inferenceStatus.style.color = 'green';
                    inferenceOutput.textContent = `Output file: ${result.output_path}`;
                } else {
                    inferenceStatus.textContent = result.message;
                    inferenceStatus.style.color = 'red';
                    inferenceOutput.textContent = result.message;
                }
            } catch (error) {
                inferenceStatus.textContent = `Error: ${error.message}`;
                inferenceStatus.style.color = 'red';
                inferenceOutput.textContent = `Error: ${error.message}`;
            }
        }
    }

    async function preprocessDataset() {
        const datasetPath = document.getElementById('datasetPath').value;
        const speechEncoder = document.getElementById('speechEncoder').value;
        const volAug = document.getElementById('volAug').checked;
        const numProcesses = parseInt(document.getElementById('preprocessNumProcesses').value);
        const trainingOutput = document.getElementById('trainingOutput');
        trainingOutput.textContent = 'Starting dataset preprocessing...
';

        try {
            const response = await fetch('/api/preprocess_dataset', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ dataset_path: datasetPath, speech_encoder: speechEncoder, vol_aug: volAug, num_processes: numProcesses })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value);
                chunk.split('\n\n').forEach(event => {
                    if (event.startsWith('data: ')) {
                        trainingOutput.textContent += event.substring(6) + '\n';
                        trainingOutput.scrollTop = trainingOutput.scrollHeight;
                    }
                });
            }
            trainingOutput.textContent += '\nDataset preprocessing completed.
';
        } catch (error) {
            trainingOutput.textContent += `\nError during preprocessing: ${error.message}\n`;
            trainingOutput.scrollTop = trainingOutput.scrollHeight;
        }
    }

    async function generateHubertF0() {
        const f0Predictor = document.getElementById('f0Predictor').value;
        const useDiff = document.getElementById('useDiff').checked;
        const numProcesses = parseInt(document.getElementById('hubertNumProcesses').value);
        const trainingOutput = document.getElementById('trainingOutput');
        trainingOutput.textContent += '\nStarting Hubert & F0 generation...
';

        try {
            const response = await fetch('/api/generate_hubert_f0', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ f0_predictor: f0Predictor, use_diff: useDiff, num_processes: numProcesses })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value);
                chunk.split('\n\n').forEach(event => {
                    if (event.startsWith('data: ')) {
                        trainingOutput.textContent += event.substring(6) + '\n';
                        trainingOutput.scrollTop = trainingOutput.scrollHeight;
                    }
                });
            }
            trainingOutput.textContent += '\nHubert & F0 generation completed.
';
        } catch (error) {
            trainingOutput.textContent += `\nError during Hubert & F0 generation: ${error.message}\n`;
            trainingOutput.scrollTop = trainingOutput.scrollHeight;
        }
    }

    async function trainModel() {
        const configPath = document.getElementById('trainConfigPath').value;
        const modelName = document.getElementById('modelName').value;
        const trainingOutput = document.getElementById('trainingOutput');
        trainingOutput.textContent += '\nStarting model training...
';

        try {
            const response = await fetch('/api/train_model', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ config_path: configPath, model_name: modelName })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value);
                chunk.split('\n\n').forEach(event => {
                    if (event.startsWith('data: ')) {
                        trainingOutput.textContent += event.substring(6) + '\n';
                        trainingOutput.scrollTop = trainingOutput.scrollHeight;
                    }
                });
            }
            trainingOutput.textContent += '\nModel training completed.
';
        } catch (error) {
            trainingOutput.textContent += `\nError during model training: ${error.message}\n`;
            trainingOutput.scrollTop = trainingOutput.scrollHeight;
        }
    }
</script>
</body>
</html>
    '''


if __name__ == "__main__":
    t = threading.Thread(target=start_server)
    t.daemon = True
    t.start()
    webview.create_window("TitanClone - Voice Conversion", "http://127.0.0.1:5000", width=1000, height=800)
    webview.start()
