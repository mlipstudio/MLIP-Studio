from orb_models.forcefield import pretrained
MACE_MODELS = {
    "MACE MPA Medium": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model",
    "MACE OMAT Medium": "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-medium.model",
    "MACE OMAT Small": "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-small.model",
    "MACE MATPES r2SCAN Medium": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-r2scan-omat-ft.model",
    "MACE MATPES PBE Medium": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model",
    "MACE MP 0a Small": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model",
    "MACE MP 0a Medium": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model",
    "MACE MP 0a Large": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2024-01-07-mace-128-L2_epoch-199.model",
    "MACE MP 0b Small": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b/mace_agnesi_small.model",
    "MACE MP 0b Medium": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b/mace_agnesi_medium.model",
    "MACE MP 0b2 Small": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b2/mace-small-density-agnesi-stress.model", # Corrected name from original code
    "MACE MP 0b2 Medium": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b2/mace-medium-density-agnesi-stress.model",
    "MACE MP 0b2 Large": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b2/mace-large-density-agnesi-stress.model",
    "MACE MP 0b3 Medium": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model",
    "MACE ANI-CC Large (500k)": "https://github.com/ACEsuit/mace/raw/main/mace/calculators/foundations_models/ani500k_large_CC.model",
    "MACE OMOL-0 XL 4M": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_omol_0/mace-omol-0-extra-large-4M.model",
    "MACE OMOL-0 XL 1024": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_omol_0/MACE-omol-0-extra-large-1024.model",
    "MACE OFF 23 Large": "https://github.com/ACEsuit/mace-off/raw/main/mace_off23/MACE-OFF23_large.model",
    "MACE OFF 23 Medium": "https://github.com/ACEsuit/mace-off/raw/main/mace_off23/MACE-OFF23_medium.model",
    "MACE OFF 23 Small": "https://github.com/ACEsuit/mace-off/raw/main/mace_off23/MACE-OFF23_small.model",
    "MACE OFF 24 Medium": "https://github.com/ACEsuit/mace-off/raw/main/mace_off24/MACE-OFF24_medium.model",
    "MACE POLAR 1 S": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_polar_1/MACE-POLAR-1-S.model",
    "MACE POLAR 1 M": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_polar_1/MACE-POLAR-1-M.model",
    "MACE POLAR 1 L": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_polar_1/MACE-POLAR-1-L.model"
}
MACE_CITATIONS = {
    # --- MACE-MP (Materials Project) Models ---
    "MACE MP 0a Small": "**Model:** Batatia et al., *J. Chem. Phys.* 163, 184110 (2025) [arXiv:2312.15211]  \n**Data:** Jain et al., *APL Mater.* 1, 011002 (2013) (Materials Project)",
    "MACE MP 0a Medium": "**Model:** Batatia et al., *J. Chem. Phys.* 163, 184110 (2025) [arXiv:2312.15211]  \n**Data:** Jain et al., *APL Mater.* 1, 011002 (2013) (Materials Project)",
    "MACE MP 0a Large": "**Model:** Batatia et al., *J. Chem. Phys.* 163, 184110 (2025) [arXiv:2312.15211]  \n**Data:** Jain et al., *APL Mater.* 1, 011002 (2013) (Materials Project)",
    "MACE MP 0b Small": "**Model:** Batatia et al., *J. Chem. Phys.* 163, 184110 (2025) [arXiv:2312.15211]  \n**Data:** Jain et al., *APL Mater.* 1, 011002 (2013) (Materials Project)",
    "MACE MP 0b Medium": "**Model:** Batatia et al., *J. Chem. Phys.* 163, 184110 (2025) [arXiv:2312.15211]  \n**Data:** Jain et al., *APL Mater.* 1, 011002 (2013) (Materials Project)",
    "MACE MP 0b2 Small": "**Model:** Batatia et al., *J. Chem. Phys.* 163, 184110 (2025) [arXiv:2312.15211]  \n**Data:** Jain et al., *APL Mater.* 1, 011002 (2013) (Materials Project)",
    "MACE MP 0b2 Medium": "**Model:** Batatia et al., *J. Chem. Phys.* 163, 184110 (2025) [arXiv:2312.15211]  \n**Data:** Jain et al., *APL Mater.* 1, 011002 (2013) (Materials Project)",
    "MACE MP 0b2 Large": "**Model:** Batatia et al., *J. Chem. Phys.* 163, 184110 (2025) [arXiv:2312.15211]  \n**Data:** Jain et al., *APL Mater.* 1, 011002 (2013) (Materials Project)",
    "MACE MP 0b3 Medium": "**Model:** Batatia et al., *J. Chem. Phys.* 163, 184110 (2025) [arXiv:2312.15211]  \n**Data:** Jain et al., *APL Mater.* 1, 011002 (2013) (Materials Project)",
    
    # --- MACE-MPA (Materials Project Augmented) ---
    "MACE MPA Medium": "**Model:** Batatia et al., *J. Chem. Phys.* 163, 184110 (2025)  \n**Data:** Jain et al., *APL Mater.* 1, 011002 (2013) (Materials Project)",

    # --- MACE-OMAT (Open Materials) ---
    "MACE OMAT Medium": "**Model:** Batatia et al., *arXiv:2510.25380* (2025) (Cross Learning/OMAT)  \n**Data:** OMat24 Dataset (Meta FAIR), *arXiv:2410.12771* (2024)",
    "MACE OMAT Small": "**Model:** Batatia et al., *arXiv:2510.25380* (2025) (Cross Learning/OMAT)  \n**Data:** OMat24 Dataset (Meta FAIR), *arXiv:2410.12771* (2024)",

    # --- MACE-OMOL (Open Molecules) ---
    "MACE OMOL-0 XL 4M": "**Model:** Batatia et al., *arXiv:2510.24063* (2025) (MACE-OMol-0)  \n**Data:** OMol24/25 Dataset (Meta FAIR)",
    "MACE OMOL-0 XL 1024": "**Model:** Batatia et al., *arXiv:2510.24063* (2025) (MACE-OMol-0)  \n**Data:** OMol24/25 Dataset (Meta FAIR)",

    # --- MACE-MATPES (PES Finetuned) ---
    "MACE MATPES r2SCAN Medium": "**Model:** Batatia et al., *J. Chem. Phys.* 163, 184110 (2025)  \n**Data:** MatPES/MP-ALOE (r2SCAN), Kuner et al., *npj Comput. Mater.* 11, 1 (2025)",
    "MACE MATPES PBE Medium": "**Model:** Batatia et al., *J. Chem. Phys.* 163, 184110 (2025)  \n**Data:** MatPES/Materials Project (PBE)",

    # --- MACE-OFF (Open Force Field) ---
    "MACE OFF 23 Small": "**Model:** Kovács et al., *J. Chem. Theory Comput.* (2024) [arXiv:2312.15211]  \n**Data:** Eastman et al., *J. Chem. Theory Comput.* 19, 209 (2023) (SPICE)",
    "MACE OFF 23 Medium": "**Model:** Kovács et al., *J. Chem. Theory Comput.* (2024) [arXiv:2312.15211]  \n**Data:** Eastman et al., *J. Chem. Theory Comput.* 19, 209 (2023) (SPICE)",
    "MACE OFF 23 Large": "**Model:** Kovács et al., *J. Chem. Theory Comput.* (2024) [arXiv:2312.15211]  \n**Data:** Eastman et al., *J. Chem. Theory Comput.* 19, 209 (2023) (SPICE)",
    "MACE OFF 24 Medium": "**Model:** Kovács et al., *arXiv:2312.15211* (updated 2024)  \n**Data:** Eastman et al., *J. Chem. Theory Comput.* 19, 209 (2023) (SPICE 2.0)",

    # --- MACE ANI-CC ---
    "MACE ANI-CC Large (500k)": "**Model:** Batatia et al., *NeurIPS* (2022) (MACE Architecture)  \n**Data:** Smith et al., *Nat. Commun.* 11, 2965 (2020) (ANI-1ccx)",
    # --- MACE POLAR ---
    "MACE POLAR 1 S": "Batatia, Ilyes, et al. *MACE-POLAR-1: A Polarisable Electrostatic Foundation Model for Molecular Chemistry.* arXiv preprint arXiv:2602.19411 (2026).",
    "MACE POLAR 1 M": "Batatia, Ilyes, et al. *MACE-POLAR-1: A Polarisable Electrostatic Foundation Model for Molecular Chemistry.* arXiv preprint arXiv:2602.19411 (2026).",
    "MACE POLAR 1 L": "Batatia, Ilyes, et al. *MACE-POLAR-1: A Polarisable Electrostatic Foundation Model for Molecular Chemistry.* arXiv preprint arXiv:2602.19411 (2026)."
}

FAIRCHEM_MODELS = {
    "UMA Small 1.2": "uma-s-1p2",
    "UMA Small 1.1": "uma-s-1p1",
    # "UMA Small 1": "uma-s-1", # No longer available
    "ESEN MD Direct All OMOL": "esen-md-direct-all-omol",
    "ESEN SM Conserving All OMOL": "esen-sm-conserving-all-omol",
    "ESEN SM Direct All OMOL": "esen-sm-direct-all-omol"
}
FAIRCHEM_CITATIONS = {
    "UMA Small 1.2": "Wood, Brandon M., et al. *Uma: A family of universal models for atoms.* arXiv preprint arXiv:2506.23971 (2025).",
    "UMA Small 1.1": "Wood, Brandon M., et al. *Uma: A family of universal models for atoms.* arXiv preprint arXiv:2506.23971 (2025).",
    # "UMA Small 1": "uma-s-1", # No longer available
    "ESEN MD Direct All OMOL": "Fu, Xiang, et al. *Learning smooth and expressive interatomic potentials for physical property prediction.* arXiv preprint arXiv:2502.12147 (2025).",
    "ESEN SM Conserving All OMOL": "Fu, Xiang, et al. *Learning smooth and expressive interatomic potentials for physical property prediction.* arXiv preprint arXiv:2502.12147 (2025).",
    "ESEN SM Direct All OMOL": "Fu, Xiang, et al. *Learning smooth and expressive interatomic potentials for physical property prediction.* arXiv preprint arXiv:2502.12147 (2025)."
}
# Define the available ORB models
ORB_MODELS = {
    "V3 OMOL Conservative": pretrained.orb_v3_conservative_omol,
    "V3 OMOL Direct": pretrained.orb_v3_direct_omol,
    "V3 OMAT Conservative (inf)": pretrained.orb_v3_conservative_inf_omat,
    "V3 OMAT Conservative (20)": pretrained.orb_v3_conservative_20_omat,
    "V3 OMAT Direct (inf)": pretrained.orb_v3_direct_inf_omat,
    "V3 OMAT Direct (20)": pretrained.orb_v3_direct_20_omat,
    "V3 MPA Conservative (inf)": pretrained.orb_v3_conservative_inf_mpa,
    "V3 MPA Conservative (20)": pretrained.orb_v3_conservative_20_mpa,
    "V3 MPA Direct (inf)": pretrained.orb_v3_direct_inf_mpa,
    "V3 MPA Direct (20)": pretrained.orb_v3_direct_20_mpa,
}
ORB_CITATIONS = {
    "V3 OMOL Conservative": "Rhodes, Benjamin, et al. *Orb-v3: atomistic simulation at scale.* arXiv preprint arXiv:2504.06231 (2025).",
    "V3 OMOL Direct": "Rhodes, Benjamin, et al. *Orb-v3: atomistic simulation at scale.* arXiv preprint arXiv:2504.06231 (2025).",
    "V3 OMAT Conservative (inf)": "Rhodes, Benjamin, et al. *Orb-v3: atomistic simulation at scale.* arXiv preprint arXiv:2504.06231 (2025).",
    "V3 OMAT Conservative (20)": "Rhodes, Benjamin, et al. *Orb-v3: atomistic simulation at scale.* arXiv preprint arXiv:2504.06231 (2025).",
    "V3 OMAT Direct (inf)": "Rhodes, Benjamin, et al. *Orb-v3: atomistic simulation at scale.* arXiv preprint arXiv:2504.06231 (2025).",
    "V3 OMAT Direct (20)": "Rhodes, Benjamin, et al. *Orb-v3: atomistic simulation at scale.* arXiv preprint arXiv:2504.06231 (2025).",
    "V3 MPA Conservative (inf)": "Rhodes, Benjamin, et al. *Orb-v3: atomistic simulation at scale.* arXiv preprint arXiv:2504.06231 (2025).",
    "V3 MPA Conservative (20)": "Rhodes, Benjamin, et al. *Orb-v3: atomistic simulation at scale.* arXiv preprint arXiv:2504.06231 (2025).",
    "V3 MPA Direct (inf)": "Rhodes, Benjamin, et al. *Orb-v3: atomistic simulation at scale.* arXiv preprint arXiv:2504.06231 (2025).",
    "V3 MPA Direct (20)": "Rhodes, Benjamin, et al. *Orb-v3: atomistic simulation at scale.* arXiv preprint arXiv:2504.06231 (2025).",
}
# Define the available MatterSim models
MATTERSIM_MODELS = {
    "V1 SMALL": "MatterSim-v1.0.0-1M.pth",
    "V1 LARGE": "MatterSim-v1.0.0-5M.pth"
}
MATTERSIM_CITATIONS = {
    "V1 SMALL": "Yang, Han, et al. *Mattersim: A deep learning atomistic model across elements, temperatures and pressures.* arXiv preprint arXiv:2405.04967 (2024).",
    "V1 LARGE": "Yang, Han, et al. *Mattersim: A deep learning atomistic model across elements, temperatures and pressures.* arXiv preprint arXiv:2405.04967 (2024)."
}
# Define the available UPET models
UPET_MODELS = {
    # PET-MAD - materials and molecules
    "PET-MAD-XS-V1.5.0": "pet-mad-xs",
    "PET-MAD-S-V1.5.0": "pet-mad-s",
    "PET-MAD-S-V1.1.0": "pet-mad-s",
    "PET-MAD-S-V1.0.2": "pet-mad-s",

    # PET-OAM (PBE Materials Project) - materials
    "PET-OAM-L-V0.1.0": "pet-oam-l",
    # "PET-OAM-XL-V0.1.0": "pet-oam-xl",



    # PET-OMat (PBE) - materials
    "PET-OMAT-XS-V1.0.0": "pet-omat-xs",
    "PET-OMAT-S-V1.0.0": "pet-omat-s",
    "PET-OMAT-M-V1.0.0": "pet-omat-m",
    "PET-OMAT-L-V1.0.0": "pet-omat-l",
    # "PET-OMAT-XL-V1.0.0": "pet-omat-xl",

    # PET-OMATPES (r2SCAN) - materials
    "PET-OMATPES-L-V0.1.0": "pet-omatpes-l",

    # PET-SPICE (wB97M-D3) - molecules
    "PET-SPICE-S-V0.2.0": "pet-spice-s",
    "PET-SPICE-L-V0.2.0": "pet-spice-l",

    "PET-MAD-DOS": "pet-mad-dos",

    "PET-OMAD-XS-V1.0.0": "pet-omad-xs",
    "PET-OMAD-S-V1.0.0": "pet-omad-s",
    "PET-OMAD-L-V0.1.0": "pet-omad-l",
}

UPET_MODELS_VERSIONS = {
    # PET-MAD - materials and molecules
    "PET-MAD-XS-V1.5.0": "1.5.0",
    "PET-MAD-S-V1.5.0": "1.5.0",
    "PET-MAD-S-V1.1.0": "1.5.0",
    "PET-MAD-S-V1.0.2": "1.0.2",

    # PET-OAM (PBE Materials Project) - materials
    "PET-OAM-L-V0.1.0": "0.1.0",
    "PET-OAM-XL-V0.1.0": "0.1.0",



    # PET-OMat (PBE) - materials
    "PET-OMAT-XS-V1.0.0": "1.0.0",
    "PET-OMAT-S-V1.0.0": "1.0.0",
    "PET-OMAT-M-V1.0.0": "1.0.0",
    "PET-OMAT-L-V1.0.0": "1.0.0",
    "PET-OMAT-XL-V1.0.0": "1.0.0",

    # PET-OMATPES (r2SCAN) - materials
    "PET-OMATPES-L-V0.1.0": "0.1.0",

    # PET-SPICE (wB97M-D3) - molecules
    "PET-SPICE-S-V0.2.0": "0.2.0",
    "PET-SPICE-L-V0.2.0": "0.2.0",

    "PET-MAD-DOS": "pet-mad-dos",

    "PET-OMAD-XS-V1.0.0": "1.0.0",
    "PET-OMAD-S-V1.0.0": "1.0.0",
    "PET-OMAD-L-V0.1.0": "0.1.0",
}
UPET_CITATIONS = {
    # PET-MAD - materials and molecules
    "PET-MAD-XS-V1.5.0": "Malosso, Cesare, et al. *High-quality, high-information datasets for universal atomistic machine learning.* arXiv preprint arXiv:2603.02089 (2026).",
    "PET-MAD-S-V1.5.0": "Malosso, Cesare, et al. *High-quality, high-information datasets for universal atomistic machine learning.* arXiv preprint arXiv:2603.02089 (2026).",
    "PET-MAD-S-V1.1.0": "Mazitov, Arslan, et al. *PET-MAD as a lightweight universal interatomic potential for advanced materials modeling.* Nature Communications 16.1 (2025): 10653.",
    "PET-MAD-S-V1.0.2": "Mazitov, Arslan, et al. *PET-MAD as a lightweight universal interatomic potential for advanced materials modeling.* Nature Communications 16.1 (2025): 10653.",

    # PET-OAM (PBE Materials Project) - materials
    "PET-OAM-L-V0.1.0": "Mazitov, Arslan, et al. *PET-MAD as a lightweight universal interatomic potential for advanced materials modeling.* Nature Communications 16.1 (2025): 10653.",
    # "PET-OAM-XL-V0.1.0": "pet-oam-xl",



    # PET-OMat (PBE) - materials
    "PET-OMAT-XS-V1.0.0": "Mazitov, Arslan, et al. *PET-MAD as a lightweight universal interatomic potential for advanced materials modeling.* Nature Communications 16.1 (2025): 10653.",
    "PET-OMAT-S-V1.0.0": "Mazitov, Arslan, et al. *PET-MAD as a lightweight universal interatomic potential for advanced materials modeling.* Nature Communications 16.1 (2025): 10653.",
    "PET-OMAT-M-V1.0.0": "Mazitov, Arslan, et al. *PET-MAD as a lightweight universal interatomic potential for advanced materials modeling.* Nature Communications 16.1 (2025): 10653.",
    "PET-OMAT-L-V1.0.0": "Mazitov, Arslan, et al. *PET-MAD as a lightweight universal interatomic potential for advanced materials modeling.* Nature Communications 16.1 (2025): 10653.",
    # "PET-OMAT-XL-V1.0.0": "pet-omat-xl",

    # PET-OMATPES (r2SCAN) - materials
    "PET-OMATPES-L-V0.1.0": "Mazitov, Arslan, et al. *PET-MAD as a lightweight universal interatomic potential for advanced materials modeling.* Nature Communications 16.1 (2025): 10653.",

    # PET-SPICE (wB97M-D3) - molecules
    "PET-SPICE-S-V0.2.0": "Mazitov, Arslan, et al. *PET-MAD as a lightweight universal interatomic potential for advanced materials modeling.* Nature Communications 16.1 (2025): 10653.",
    "PET-SPICE-L-V0.2.0": "Mazitov, Arslan, et al. *PET-MAD as a lightweight universal interatomic potential for advanced materials modeling.* Nature Communications 16.1 (2025): 10653.",

    "PET-MAD-DOS": "How, Wei Bin, et al. *A universal machine learning model for the electronic density of states.* Digital Discovery (2026).",

    "PET-OMAD-XS-V1.0.0": "Mazitov, Arslan, et al. *PET-MAD as a lightweight universal interatomic potential for advanced materials modeling.* Nature Communications 16.1 (2025): 10653.",
    "PET-OMAD-S-V1.0.0": "Mazitov, Arslan, et al. *PET-MAD as a lightweight universal interatomic potential for advanced materials modeling.* Nature Communications 16.1 (2025): 10653.",
    "PET-OMAD-L-V0.1.0": "Mazitov, Arslan, et al. *PET-MAD as a lightweight universal interatomic potential for advanced materials modeling.* Nature Communications 16.1 (2025): 10653.",
}
SEVEN_NET_MODELS = {
    "7net-0": "7net-0",
    "7net-l3i5": "7net-l3i5",
    "7net-omat": "7net-omat",
    "7net-mf-ompa": "7net-mf-ompa",
    "7net-omni": "7net-omni",
    # "7net-omni-i8": "7net-omni-i8",
    # "7net-omni-i12": "7net-omni-i12",
}
SEVEN_NET_CITATIONS = {
    "7net-0": "Park, Yutack, et al. *Scalable parallel algorithm for graph neural network interatomic potentials in molecular dynamics simulations.* Journal of chemical theory and computation 20.11 (2024): 4857-4868.",
    "7net-l3i5": "Park, Yutack, et al. *Scalable parallel algorithm for graph neural network interatomic potentials in molecular dynamics simulations.* Journal of chemical theory and computation 20.11 (2024): 4857-4868.",
    "7net-omat": "Park, Yutack, et al. *Scalable parallel algorithm for graph neural network interatomic potentials in molecular dynamics simulations.* Journal of chemical theory and computation 20.11 (2024): 4857-4868.",
    "7net-mf-ompa": "Kim, Jaesun, et al. *Data-efficient multifidelity training for high-fidelity machine learning interatomic potentials.* Journal of the American Chemical Society 147.1 (2024): 1042-1054.",
    "7net-omni": "Kim, Jaesun, et al. *Optimizing cross-domain transfer for universal machine learning interatomic potentials.* Nature Communications (2026).",
    # "7net-omni-i8": "Kim, Jaesun, et al. *Optimizing cross-domain transfer for universal machine learning interatomic potentials.* Nature Communications (2026).",
    # "7net-omni-i12": "Kim, Jaesun, et al. *Optimizing cross-domain transfer for universal machine learning interatomic potentials.* Nature Communications (2026).",
}


# Dictionary of sample structures
SAMPLE_STRUCTURES = {
    "Water": "H2O.xyz",
    "Methane": "CH4.xyz",
    "Ethane": "C2H6.xyz",
    "Benzene": "C6H6.xyz",
    "Fulvene": "Fulvene.xyz",
    "Caffeine": "caffeine.xyz",
    "Ibuprofen": "ibuprofen.xyz",
    "C60": "C60.xyz",
    "Aspirin": "aspirin.xyz",
    "Taxol": "Taxol.xyz",
    "Valinomycin": "Valinomycin.xyz",
    "Olestra": "Olestra.xyz",
    "Ubiquitin": "Ubiquitin.xyz",
    "Silicon": "Si.cif",
    "Copper": "Cu.cif",
    "Molybdenum": "Mo.cif",
    "Al2O3 (bulk)": "Al2O3.cif",
    "MoS2 (bulk)": "MoS2.cif",
    "MoSe2 (bulk)": "MoSe2.cif",
    "Liquid water 64 (bulk)": "water_64.extxyz",
    "Al2O3 (0001) Surface": "Al2O3_0001.cif",
    "hBN Monolayer (4x4)": "hBN_monolayer_4x4_supercell.extxyz",
    "Graphene Monolayer (4x4)": "graphene_monolayer_4x4_supercell.extxyz",
    "Cu(111) Surface": "Cu111_slab.cif",
    "CO on Cu(111)": "CO_on_Cu111.xyz",
}