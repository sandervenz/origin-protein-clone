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

# Import functions from separate files
from llm import get_llm_response
from denovo import generate_protein

# Apply patch for asyncio in Streamlit's environment
nest_asyncio.apply()

st.set_page_config(layout="wide", page_title="Universa AI-Origin")

# ==============================================================================
# HELPER FUNCTIONS (Imported or Local)
# ==============================================================================

def fetch_pdb_from_esmfold(sequence):
    """Fetches a PDB structure from the ESMFold API."""
    try:
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        res = requests.post("https://api.esmatlas.com/foldSequence/v1/pdb/", headers=headers, data=sequence, timeout=90)
        res.raise_for_status()
        return res.text
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch structure from ESMFold: {e}")
        return None

def relax_protein_structure(pdb_str, settings):
    """Relaxes a PDB structure using Amber (if AlphaFold is installed)."""
    try:
        from alphafold.common import protein
        from alphafold.relax import relax
    except ImportError:
        st.warning("AlphaFold library not found. Skipping relaxation process.")
        return pdb_str

    relaxer = relax.AmberRelaxation(
        max_iterations=settings['max_iterations'],
        tolerance=settings['tolerance'],
        stiffness=settings['stiffness'],
        exclude_residues=[],
        max_outer_iterations=3,
        use_gpu=settings['use_gpu']
    )
    prot_obj = protein.from_pdb_string(pdb_str)
    relaxed_pdb_str, _, _ = relaxer.process(prot=prot_obj)
    return relaxed_pdb_str

def view_structure_with_py3dmol(pdb_str, settings):
    """Creates a 3D view of the PDB string using py3Dmol."""
    view = py3Dmol.view(width=800, height=600)
    view.addModel(pdb_str, 'pdb')
    
    # Set main cartoon style based on color scheme
    if settings['color_scheme'] == "lDDT":
        view.setStyle({'cartoon': {'colorscheme': {'prop': 'b', 'gradient': 'roygb', 'min': 50, 'max': 90}}})
    else: # rainbow
        view.setStyle({'cartoon': {'color': 'spectrum'}})
    
    # MODIFIED: Add backbone and sidechain views if selected
    if settings.get('show_backbone', False):
        view.addStyle(
            {'atom': ['C', 'O', 'N', 'CA']},
            {'stick': {'colorscheme': 'WhiteCarbon', 'radius': 0.2}}
        )
    if settings.get('show_sidechains', False):
        view.addStyle(
            {'resn': ["ALA", "GLY"], 'invert': True}, # Show sidechains for all except ALA and GLY
            {'stick': {'colorscheme': 'WhiteCarbon', 'radius': 0.2}}
        )
    
    view.setBackgroundColor('#1E1E3F')
    view.zoomTo()
    return view

def plot_plddt(pdb_str):
    """Plots the pLDDT scores from a PDB file's B-factor column."""
    parser = PDBParser(QUIET=True)
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".pdb") as tmp:
        tmp.write(pdb_str)
        pdb_path = tmp.name
    
    try:
        structure = parser.get_structure("P", pdb_path)
        plddt_scores = [atom.get_bfactor() for atom in structure.get_atoms() if atom.get_id() == 'CA']
        
        fig, ax = plt.subplots(facecolor='#2a2a4e')
        ax.plot(plddt_scores, color='#00aaff')
        ax.set_title("pLDDT Score per Residue", color='white')
        ax.set_xlabel("Residue Index", color='white')
        ax.set_ylabel("pLDDT Score", color='white')
        ax.grid(True, color='gray', linestyle='--')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        st.pyplot(fig)
    finally:
        os.remove(pdb_path)

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================

DEFAULTS = {
    'logged_in': False,
    'selected_modules': ["Prompt Refinement"],
    'auto_mode': True,
    'user_prompt': "",
    'refined_prompt': "",
    'generated_sequences_df': pd.DataFrame(),
    'selected_sequence': "",
    'raw_pdb': "",
    'relaxed_pdb': "",
    'run_refine': False,
    'run_generate': False,
    'run_structure': False,
    'num_sequences': 5,
    'vis_settings': {
        'max_iterations': 2000,
        'tolerance': 2.39,
        'stiffness': 10.0,
        'use_gpu': False,
        'color_scheme': 'rainbow',
        'display_option': 'Raw PDB',
        'show_backbone': False,
        'show_sidechains': False,
    }
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ==============================================================================
# WORKFLOW LOGIC FUNCTIONS
# ==============================================================================

def execute_refine_prompt():
    """Calls the LLM to refine the user's prompt."""
    st.toast('🚀 Refining prompt...')
    with st.spinner("Refining prompt with AI..."):
        st.session_state.refined_prompt = asyncio.run(get_llm_response(st.session_state.user_prompt))
    
    if st.session_state.auto_mode and "Sequence Generator" in st.session_state.selected_modules:
        st.session_state.run_generate = True

def execute_generate_sequence():
    """Generates sequences based on the refined prompt."""
    st.toast('🧬 Generating sequences...')
    prompt_to_use = st.session_state.refined_prompt or st.session_state.user_prompt
    with st.spinner(f"Generating {st.session_state.num_sequences} protein sequences..."):
        st.session_state.generated_sequences_df = generate_protein(prompt_to_use, st.session_state.num_sequences)
    
    if not st.session_state.generated_sequences_df.empty:
        st.session_state.selected_sequence = st.session_state.generated_sequences_df.iloc[0]['ProteinSequence']

    if st.session_state.auto_mode and "Structure Visualisation" in st.session_state.selected_modules:
        st.session_state.run_structure = True

def execute_generate_structure():
    """Generates and relaxes the 3D structure for the selected sequence."""
    st.toast('🔬 Predicting 3D structure...')
    with st.spinner("Fetching 3D structure from ESMFold..."):
        st.session_state.raw_pdb = fetch_pdb_from_esmfold(st.session_state.selected_sequence)
    
    if st.session_state.raw_pdb:
        with st.spinner("Relaxing structure (optional)..."):
            st.session_state.relaxed_pdb = relax_protein_structure(st.session_state.raw_pdb, st.session_state.vis_settings)

def reset_workflow_state():
    """Resets only the inputs and outputs of the workflow, not settings."""
    st.session_state.user_prompt = ""
    st.session_state.refined_prompt = ""
    st.session_state.generated_sequences_df = pd.DataFrame()
    st.session_state.selected_sequence = ""
    st.session_state.raw_pdb = ""
    st.session_state.relaxed_pdb = ""
    st.toast("✨ Workflow has been reset!")

# ==============================================================================
# UI RENDERING
# ==============================================================================

# --- LOGIN PAGE ---
if not st.session_state.logged_in:
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.title("UNIVERSA AI-ORIGIN")
    st.header("AI DRIVEN PROTEIN DESIGN")
    
    username = st.text_input("USER NAME", key="login_user")
    if st.button("LOGIN"):
        if username:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Username cannot be empty.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- MAIN APPLICATION ---
else:
    # Check for and execute workflow triggers
    if st.session_state.run_refine:
        st.session_state.run_refine = False
        execute_refine_prompt()
    if st.session_state.run_generate:
        st.session_state.run_generate = False
        execute_generate_sequence()
    if st.session_state.run_structure:
        st.session_state.run_structure = False
        execute_generate_structure()

    # --- Application Header ---
    st.title("AI Driven Protein Design")
    st.markdown("---")

    # --- Workspace Configuration ---
    st.subheader("Workspace Configuration")
    
    cols = st.columns(5)
    selections = {
        "Prompt Refinement": cols[0].checkbox("Prompt Refinement", value=("Prompt Refinement" in st.session_state.selected_modules)),
        "Sequence Generator": cols[1].checkbox("Sequence Generator", value=("Sequence Generator" in st.session_state.selected_modules)),
        "Structure Visualisation": cols[2].checkbox("Structure Visualisation", value=("Structure Visualisation" in st.session_state.selected_modules))
    }
    st.session_state.selected_modules = [name for name, selected in selections.items() if selected]
    
    with cols[3]:
        st.session_state.auto_mode = st.toggle("Auto Mode", value=st.session_state.auto_mode, help="If active, all selected modules will run automatically in a chain.")

    with cols[4]:
        if st.button("Reset 🔄", on_click=reset_workflow_state, help="Clear all inputs and outputs."):
            pass
    
    st.markdown("---")

    # --- Render Modules Based on Selection ---
    
    # 1. PROMPT REFINEMENT MODULE
    if "Prompt Refinement" in st.session_state.selected_modules:
        with st.container(border=False):
            st.markdown('<div class="module-container">', unsafe_allow_html=True)
            st.header("1. Prompt Refinement")
            st.session_state.user_prompt = st.text_area("Enter your protein design idea or goal:", value=st.session_state.user_prompt, height=100, key="prompt_input")
            
            if st.button("Refine Prompt"):
                if st.session_state.user_prompt:
                    st.session_state.run_refine = True
                    st.rerun()
                else:
                    st.warning("Please enter a prompt first.")

            st.text_area("AI Refinement Result:", value=st.session_state.refined_prompt, height=150, disabled=True, key="prompt_output")
            st.markdown('</div>', unsafe_allow_html=True)

    # 2. SEQUENCE GENERATOR MODULE
    if "Sequence Generator" in st.session_state.selected_modules:
        with st.container(border=False):
            st.markdown('<div class="module-container">', unsafe_allow_html=True)
            
            header_cols = st.columns([1, 0.1])
            with header_cols[0]:
                st.header("2. Sequence Generator")
            with header_cols[1]:
                if st.button("⚙️", key="seq_settings_btn"):
                    st.session_state.show_seq_settings = True
            
            if st.session_state.get("show_seq_settings", False):
                @st.dialog("Sequence Generator Settings")
                def seq_settings_dialog():
                    st.session_state.num_sequences = st.number_input("Number of sequences to generate", min_value=1, max_value=100, value=st.session_state.num_sequences)
                    if st.button("Close", key="close_seq_settings"):
                        st.session_state.show_seq_settings = False
                        st.rerun()
                seq_settings_dialog()

            prompt_for_gen = st.text_area("Prompt for Generator:", value=st.session_state.refined_prompt, height=150, disabled=st.session_state.auto_mode, key="seq_input")
            
            if not st.session_state.auto_mode:
                 if st.button("Generate Sequences"):
                    if prompt_for_gen:
                        st.session_state.refined_prompt = prompt_for_gen
                        st.session_state.run_generate = True
                        st.rerun()
                    else:
                        st.warning("Prompt cannot be empty.")
            
            st.subheader("Generated Sequences")
            if not st.session_state.generated_sequences_df.empty:
                options = [f"Score {row.ProtrekScore:.2f} - {row.ProteinSequence[:30]}..." for _, row in st.session_state.generated_sequences_df.iterrows()]
                selected_option = st.selectbox("Select a sequence:", options, key="seq_selector")
                if selected_option:
                    selected_idx = options.index(selected_option)
                    st.session_state.selected_sequence = st.session_state.generated_sequences_df.iloc[selected_idx]['ProteinSequence']
            
            st.text_area("Selected Sequence:", value=st.session_state.selected_sequence, height=100, disabled=True, key="seq_output")
            st.markdown('</div>', unsafe_allow_html=True)

    # 3. STRUCTURE VISUALISATION MODULE
    if "Structure Visualisation" in st.session_state.selected_modules:
        with st.container(border=False):
            st.markdown('<div class="module-container">', unsafe_allow_html=True)

            header_cols = st.columns([1, 0.1])
            with header_cols[0]:
                 st.header("3. Structure Visualisation")
            with header_cols[1]:
                if st.button("⚙️", key="vis_settings_btn"):
                    st.session_state.show_vis_settings = True

            if st.session_state.get("show_vis_settings", False):
                @st.dialog("Structure Visualisation Settings")
                def vis_settings_dialog():
                    with st.expander("Amber Relaxation Settings", expanded=True):
                        st.session_state.vis_settings['max_iterations'] = st.slider("Max Iterations", 0, 5000, st.session_state.vis_settings['max_iterations'])
                        st.session_state.vis_settings['tolerance'] = st.number_input("Tolerance (kcal/mol)", value=st.session_state.vis_settings['tolerance'])
                        st.session_state.vis_settings['stiffness'] = st.number_input("Stiffness (kcal/mol A²)", value=st.session_state.vis_settings['stiffness'])
                        st.session_state.vis_settings['use_gpu'] = st.checkbox("Use GPU for relaxation", value=st.session_state.vis_settings['use_gpu'])
                    
                    with st.expander("Visualisation Settings", expanded=True):
                         st.session_state.vis_settings['color_scheme'] = st.selectbox("Color Scheme", ["rainbow", "lDDT"], index=["rainbow", "lDDT"].index(st.session_state.vis_settings['color_scheme']))
                         # ADDED: Checkboxes for backbone and sidechains
                         st.session_state.vis_settings['show_backbone'] = st.checkbox("Show Backbone", value=st.session_state.vis_settings['show_backbone'])
                         st.session_state.vis_settings['show_sidechains'] = st.checkbox("Show Sidechains", value=st.session_state.vis_settings['show_sidechains'])
                         st.session_state.vis_settings['display_option'] = st.radio("Select Structure to Display", ["Raw PDB", "Relaxed PDB"], index=["Raw PDB", "Relaxed PDB"].index(st.session_state.vis_settings['display_option']))
                    
                    if st.button("Close", key="close_vis_settings"):
                        st.session_state.show_vis_settings = False
                        st.rerun()
                vis_settings_dialog()

            seq_for_vis = st.text_area("Sequence for Visualisation:", value=st.session_state.selected_sequence, height=100, disabled=st.session_state.auto_mode, key="vis_input")

            if not st.session_state.auto_mode:
                if st.button("Predict 3D Structure"):
                    if seq_for_vis:
                        st.session_state.selected_sequence = seq_for_vis
                        st.session_state.run_structure = True
                        st.rerun()
                    else:
                        st.warning("Sequence cannot be empty.")

            if st.session_state.raw_pdb:
                st.subheader("Prediction Results")
                
                pdb_to_show = st.session_state.raw_pdb
                if st.session_state.vis_settings['display_option'] == 'Relaxed PDB' and st.session_state.relaxed_pdb:
                    pdb_to_show = st.session_state.relaxed_pdb
                
                vis_cols = st.columns(2)
                with vis_cols[0]:
                    st.write(f"3D Visualisation ({st.session_state.vis_settings['display_option']})")
                    st_py3dmol_view = view_structure_with_py3dmol(pdb_to_show, st.session_state.vis_settings)
                    st.components.v1.html(st_py3dmol_view._make_html(), height=600)
                
                with vis_cols[1]:
                    st.write("pLDDT Score Analysis (from Raw PDB)")
                    plot_plddt(st.session_state.raw_pdb)

                    st.write("Download Results")
                    st.download_button("Download Raw PDB", st.session_state.raw_pdb, file_name="raw_structure.pdb")
                    if st.session_state.relaxed_pdb:
                        st.download_button("Download Relaxed PDB", st.session_state.relaxed_pdb, file_name="relaxed_structure.pdb")
            
            st.markdown('</div>', unsafe_allow_html=True)