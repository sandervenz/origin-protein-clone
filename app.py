import streamlit as st
import pandas as pd
import requests
import py3Dmol
import matplotlib.pyplot as plt
import os
import asyncio
import nest_asyncio
import tempfile
from Bio.PDB import PDBParser
import subprocess
import glob

# Import functions from separate files
from llm import get_llm_response
from denovo import generate_protein

# Apply patch for asyncio in Streamlit's environment
nest_asyncio.apply()

st.set_page_config(layout="wide", page_title="Universa AI-Origin")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def fetch_pdb_from_esmfold(sequence):
    """Fetches a PDB structure from the ESMFold API."""
    with st.spinner("Fetching structure from ESMFold..."):
        try:
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            res = requests.post("https://api.esmatlas.com/foldSequence/v1/pdb/", headers=headers, data=sequence, timeout=120)
            res.raise_for_status()
            st.success("✅ Structure generated successfully!")
            return res.text
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch structure from ESMFold: {e}")
            return None

def relax_protein_structure(pdb_str, settings):
    """Relaxes a PDB structure using the colabfold_relax command-line tool."""
    with st.spinner("Relaxing structure with AMBER..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.pdb")
            with open(input_path, "w") as f:
                f.write(pdb_str)
            
            relax_executable = "/opt/conda/envs/protein_env/bin/colabfold_relax"
            
            command = [relax_executable]
            if settings['use_gpu']:
                command.append("--use-gpu")
            
            command.append(input_path)  
            command.append(tmpdir)     

            try:
                process = subprocess.run(
                    command, 
                    capture_output=True, 
                    text=True, 
                    check=True,
                    timeout=600  
                )
                
                output_files = glob.glob(os.path.join(tmpdir, "*_relaxed_*.pdb"))
                if not output_files:
                    st.warning("Relaxation ran, but no relaxed PDB file was found.")
                    return pdb_str
                    
                with open(output_files[0], "r") as f:
                    relaxed_pdb_str = f.read()
                st.success("✅ Relaxation complete!")
                return relaxed_pdb_str

            except subprocess.CalledProcessError as e:
                st.error("An error occurred during the AMBER relaxation process.")
                st.text_area("Relaxation Error Log:", e.stderr, height=200)
                return pdb_str 
            except Exception as e:
                st.error(f"An unexpected error occurred during relaxation: {e}")
                return pdb_str
            
def view_structure_with_py3dmol(pdb_str, settings):
    """Creates a 3D view of the PDB string using py3Dmol."""
    view = py3Dmol.view(width=800, height=500)
    view.addModel(pdb_str, 'pdb')
    if settings['color_scheme'] == "lDDT":
        view.setStyle({'cartoon': {'colorscheme': {'prop': 'b', 'gradient': 'roygb', 'min': 50, 'max': 90}}})
    else:
        view.setStyle({'cartoon': {'color': 'spectrum'}})
    if settings.get('show_backbone', False):
        view.addStyle({'atom': ['C', 'O', 'N', 'CA']}, {'stick': {'colorscheme': 'WhiteCarbon', 'radius': 0.2}})
    if settings.get('show_sidechains', False):
        view.addStyle({'resn': ["ALA", "GLY"], 'invert': True}, {'stick': {'colorscheme': 'WhiteCarbon', 'radius': 0.2}})
    view.setBackgroundColor('#1E1E3F')
    view.zoomTo()
    return view

def plot_plddt_comparison(raw_str, relaxed_str):
    """Plots a comparison of pLDDT scores for raw and relaxed structures."""
    def get_b_factors(pdb_str):
        parser = PDBParser(QUIET=True)
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".pdb") as tmp:
            tmp.write(pdb_str)
            pdb_path = tmp.name
        try:
            structure = parser.get_structure("P", pdb_path)
            return [atom.get_bfactor() for atom in structure.get_atoms() if atom.get_id() == 'CA']
        finally:
            os.remove(pdb_path)
    
    fig, ax = plt.subplots(facecolor='#2a2a4e')
    if raw_str:
        ax.plot(get_b_factors(raw_str), label="Raw pLDDT", color="#ff7f7f", linewidth=2)
    if relaxed_str:
        ax.plot(get_b_factors(relaxed_str), label="Relaxed pLDDT", color="#33ff33", linestyle="--", linewidth=2)
    
    ax.set_title("pLDDT Score Comparison (Raw vs Relaxed)", color='white')
    ax.set_xlabel("Residue Index", color='white')
    ax.set_ylabel("pLDDT Score", color='white')
    ax.legend()
    ax.grid(True, color='gray', linestyle='--')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    st.pyplot(fig)

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================

DEFAULTS = {
    'logged_in': False, 'selected_modules': ["Prompt Refinement", "Sequence Generator", "Structure Visualisation"],
    'auto_mode': True, 'user_prompt': "", 'refined_prompt': "", 'generated_sequences_df': pd.DataFrame(),
    'selected_sequence': "", 'raw_pdb': None, 'relaxed_pdb': None, 'run_refine': False, 'run_generate': False,
    'run_structure': False, 'num_sequences': 5,
    'vis_settings': {
        'max_iterations': 2000, 'tolerance': 2.39, 'stiffness': 10.0, 'use_gpu': False,
        'color_scheme': 'rainbow', 'display_option': 'Relaxed PDB', 'show_backbone': False, 'show_sidechains': False,
    }
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ==============================================================================
# WORKFLOW LOGIC FUNCTIONS (No changes in this section)
# ==============================================================================

def execute_refine_prompt():
    st.toast('🚀 Refining prompt...')
    with st.spinner("Refining prompt with AI..."):
        st.session_state.refined_prompt = asyncio.run(get_llm_response(st.session_state.user_prompt))
    if st.session_state.auto_mode and "Sequence Generator" in st.session_state.selected_modules:
        st.session_state.run_generate = True

def execute_generate_sequence():
    st.toast('🧬 Generating sequences...')
    prompt_to_use = st.session_state.refined_prompt or st.session_state.user_prompt
    with st.spinner(f"Generating {st.session_state.num_sequences} protein sequences..."):
        st.session_state.generated_sequences_df = generate_protein(prompt_to_use, st.session_state.num_sequences)
    if not st.session_state.generated_sequences_df.empty:
        st.session_state.selected_sequence = st.session_state.generated_sequences_df.iloc[0]['ProteinSequence']
    if st.session_state.auto_mode and "Structure Visualisation" in st.session_state.selected_modules:
        st.session_state.run_structure = True

def execute_generate_structure():
    st.toast('⚡ Predicting and relaxing structure...')
    raw_pdb = fetch_pdb_from_esmfold(st.session_state.selected_sequence)
    st.session_state.raw_pdb = raw_pdb
    if raw_pdb:
        st.session_state.relaxed_pdb = relax_protein_structure(raw_pdb, st.session_state.vis_settings)

def reset_workflow_state():
    keys_to_reset = ['user_prompt', 'refined_prompt', 'generated_sequences_df', 'selected_sequence', 'raw_pdb', 'relaxed_pdb']
    for key in keys_to_reset:
        st.session_state[key] = DEFAULTS.get(key)
    st.toast("✨ Workflow has been reset!")

# ==============================================================================
# UI RENDERING
# ==============================================================================

if not st.session_state.logged_in:
    # --- LOGIN PAGE ---
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.title("UNIVERSA AI-ORIGIN")
    st.header("AI DRIVEN PROTEIN DESIGN")
    username = st.text_input("USER NAME", key="login_user")
    if st.button("LOGIN"):
        if username:
            st.session_state.logged_in = True; st.rerun()
        else:
            st.error("Username cannot be empty.")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # --- Trigger Handling Logic ---
    if st.session_state.run_refine:
        st.session_state.run_refine = False; execute_refine_prompt()
    if st.session_state.run_generate:
        st.session_state.run_generate = False; execute_generate_sequence()
    if st.session_state.run_structure:
        st.session_state.run_structure = False; execute_generate_structure()

    # --- Header & Workspace Config ---
    st.title("AI Driven Protein Design")
    st.markdown("---")
    st.subheader("Workspace Configuration")
    cols = st.columns(5)
    selections = {"Prompt Refinement": cols[0].checkbox("Prompt Refinement", value=("Prompt Refinement" in st.session_state.selected_modules)),"Sequence Generator": cols[1].checkbox("Sequence Generator", value=("Sequence Generator" in st.session_state.selected_modules)),"Structure Visualisation": cols[2].checkbox("Structure Visualisation", value=("Structure Visualisation" in st.session_state.selected_modules))}
    st.session_state.selected_modules = [name for name, selected in selections.items() if selected]
    with cols[3]: st.session_state.auto_mode = st.toggle("Auto Mode", value=st.session_state.auto_mode, help="If active, all selected modules will run automatically in a chain.")
    with cols[4]: st.button("Reset 🔄", on_click=reset_workflow_state, help="Clear all inputs and outputs.")
    st.markdown("---")

    # --- Module Rendering with 2-Column Layout ---
    
    if "Prompt Refinement" in st.session_state.selected_modules:
        with st.container(border=False):
            st.markdown('<div class="module-container">', unsafe_allow_html=True)
            st.header("1. Prompt Refinement")
            left, right = st.columns(2)
            with left:
                st.session_state.user_prompt = st.text_area("Enter your protein design idea:", value=st.session_state.user_prompt, height=150, key="prompt_input")
                if st.button("Refine Prompt"):
                    if st.session_state.user_prompt:
                        st.session_state.run_refine = True; st.rerun()
                    else:
                        st.warning("Please enter a prompt first.")
            with right:
                st.text_area("AI Refinement Result:", value=st.session_state.refined_prompt, height=190, disabled=True, key="prompt_output", help="The AI-generated prompt will appear here.")
            st.markdown('</div>', unsafe_allow_html=True)

    if "Sequence Generator" in st.session_state.selected_modules:
        with st.container(border=False):
            st.markdown('<div class="module-container">', unsafe_allow_html=True)
            left, right = st.columns(2)

            with left:
                header_cols = st.columns([1, 0.1]); 
                with header_cols[0]: st.header("2. Sequence Generator")
                with header_cols[1]:
                    if st.button("⚙️", key="seq_settings_btn", help="Sequence generation settings"):
                        st.session_state.show_seq_settings = True
                
                if st.session_state.get("show_seq_settings", False):
                    @st.dialog("Sequence Generator Settings")
                    def seq_settings_dialog():
                        st.session_state.num_sequences = st.number_input("Number of sequences to generate", 1, 100, st.session_state.num_sequences)
                        if st.button("Close", key="close_seq_settings"):
                            st.session_state.show_seq_settings = False; st.rerun()
                    seq_settings_dialog()

                prompt_for_gen = st.text_area("Prompt for Generator:", value=st.session_state.refined_prompt, height=150, disabled=st.session_state.auto_mode, key="seq_input")
                if not st.session_state.auto_mode and st.button("Generate Sequences"):
                    if prompt_for_gen:
                        st.session_state.refined_prompt = prompt_for_gen; st.session_state.run_generate = True; st.rerun()
                    else:
                        st.warning("Prompt cannot be empty.")
            with right:
                st.subheader("Generated Sequences")
                if not st.session_state.generated_sequences_df.empty:
                    options = [f"Score {row.ProtrekScore:.2f} - {row.ProteinSequence[:30]}..." for _, row in st.session_state.generated_sequences_df.iterrows()]
                    selected_option = st.selectbox("Select a sequence:", options, key="seq_selector", label_visibility="collapsed")
                    if selected_option:
                        selected_idx = options.index(selected_option)
                        st.session_state.selected_sequence = st.session_state.generated_sequences_df.iloc[selected_idx]['ProteinSequence']
                st.text_area("Selected Sequence:", value=st.session_state.selected_sequence, height=100, disabled=True, key="seq_output")
            st.markdown('</div>', unsafe_allow_html=True)

    if "Structure Visualisation" in st.session_state.selected_modules:
        with st.container(border=False):
            st.markdown('<div class="module-container">', unsafe_allow_html=True)
            left, right = st.columns(2)

            with left:
                header_cols = st.columns([1, 0.1]); 
                with header_cols[0]: st.header("3. Structure Prediction")
                with header_cols[1]:
                    if st.button("⚙️", key="relax_settings_btn", help="AMBER Relaxation Settings"):
                        st.session_state.show_relax_settings = True
                
                if st.session_state.get("show_relax_settings", False):
                    @st.dialog("AMBER Relaxation Settings")
                    def relax_settings_dialog():
                        vis = st.session_state.vis_settings
                        vis['max_iterations'] = st.slider("Max Iterations", 0, 5000, vis['max_iterations'])
                        vis['tolerance'] = st.number_input("Tolerance (kcal/mol)", value=vis['tolerance'])
                        vis['stiffness'] = st.number_input("Stiffness (kcal/mol A²)", value=vis['stiffness'])
                        vis['use_gpu'] = st.checkbox("Use GPU for relaxation (if available)", value=vis['use_gpu'])
                        if st.button("Close", key="close_relax_settings"):
                            st.session_state.show_relax_settings = False; st.rerun()
                    relax_settings_dialog()

                st.session_state.selected_sequence = st.text_area("🧬 Sequence for Prediction:", value=st.session_state.selected_sequence, height=190, key="vis_input")
                
                if st.button("🛠️ Generate & Relax Structure"):
                    if st.session_state.selected_sequence:
                        st.session_state.run_structure = True; st.rerun()
                    else:
                        st.warning("Please provide a sequence first.")

            with right:
                st.subheader("Results")
                if st.session_state.raw_pdb:
                    with st.expander("🧪 Visualization Settings"):
                        vis = st.session_state.vis_settings
                        vis['display_option'] = st.radio("Display:", ["Raw PDB", "Relaxed PDB"], index=["Raw PDB", "Relaxed PDB"].index(vis['display_option']), horizontal=True)
                        vis['color_scheme'] = st.selectbox("Color Scheme:", ["rainbow", "lDDT"], index=["rainbow", "lDDT"].index(vis['color_scheme']))
                        vis['show_backbone'] = st.checkbox("Show Backbone", value=vis['show_backbone'])
                        vis['show_sidechains'] = st.checkbox("Show Sidechains", value=vis['show_sidechains'])

                    pdb_to_show = st.session_state.relaxed_pdb if vis['display_option'] == 'Relaxed PDB' and st.session_state.relaxed_pdb else st.session_state.raw_pdb
                    if pdb_to_show:
                        viewer = view_structure_with_py3dmol(pdb_to_show, st.session_state.vis_settings)
                        st.components.v1.html(viewer._make_html(), height=400)
                    
                    plot_plddt_comparison(st.session_state.raw_pdb, st.session_state.relaxed_pdb)
                    dl_col1, dl_col2 = st.columns(2)
                    with dl_col1:
                        st.download_button("Download Raw PDB", st.session_state.raw_pdb, file_name="raw_structure.pdb")
                    with dl_col2:
                        if st.session_state.relaxed_pdb:
                            st.download_button("Download Relaxed PDB", st.session_state.relaxed_pdb, file_name="relaxed_structure.pdb")
                else:
                    st.info("Output will be displayed here after prediction.")
            
            st.markdown('</div>', unsafe_allow_html=True)