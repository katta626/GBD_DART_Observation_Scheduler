from flask import Flask, request
import os, subprocess
from flask import jsonify, send_file
log_files = "/home/dsp/PDR_acquire_setup/master_obs_log"
app = Flask(__name__)

@app.route('/trigger', methods=['POST'])
def trigger_observation():
    data = request.json
    print("Received parameters:", data)
    #print(type(data))
    # Run your observation code here
    run_observation(data)
    
    return {"status": "observation started"}, 200

def run_observation(params):
    observation_script = "/home/dsp/PDR_acquire_setup/GBD_DART_lite_obs_script_V5_IQUV_FITS_09_09_2025.sh"
    #print("Running observation with:", params)
    
    obs_cmd = observation_script+" "+str(params["duration"])+" "+str(params["target"])+' '+str(params["countdown"])
    
    start_screen =  "screen -dmS "+'"'+ str(params["target"])+'"'
    end_screen   =  "screen -S "+'"'+ str(params["target"])+'"'+ " -X  quit"
    obs_trigger_cmd = 'screen -S '+ str(params["target"]) +" -X stuff "+"' "+ str(obs_cmd)+" ^M'"

    os.system(end_screen)          # To kill old screen session
    os.system(start_screen)        # To Start new screen session
    os.system(obs_trigger_cmd)     # Execute commands in new screen session
    

@app.route('/get-log', methods=['GET'])
def get_log_file():
    filename = request.args.get("filename")
    if not filename:
        return jsonify({"error": "Filename parameter missing"}), 400

    #safe_filename = os.path.basename(filename)
    log_path = log_files+"/"+filename
    
    print(filename, log_path)
    

    if os.path.exists(log_path):
        return send_file(log_path, as_attachment=True)
    else:
        return jsonify({"error": "Log file not found"}), 404
        

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)  # open to all IPs on port 5000
