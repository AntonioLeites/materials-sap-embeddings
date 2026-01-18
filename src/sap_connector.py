"""
Utilities for SAP data - both real connections and synthetic data generation
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import random


class SAPMaterialLoader:
    """
    Load material master data from SAP
    (Placeholder for real SAP connection)
    """
    
    def __init__(self, connection_params: Dict = None):
        """
        Initialize SAP connection
        
        Args:
            connection_params: SAP connection parameters
                - host, client, user, password for RFC
                - or OData endpoint URL
        """
        self.connection_params = connection_params or {}
    
    def load_materials(
        self,
        material_type: str = None,
        plant: str = None,
        limit: int = None
    ) -> pd.DataFrame:
        """
        Load materials from SAP (real implementation)
        
        TODO: Implement actual SAP connection using:
        - pyrfc (SAP NetWeaver RFC)
        - OData APIs
        - Direct HANA connection
        - CDS view extraction
        """
        raise NotImplementedError(
            "SAP connection not implemented. "
            "Use create_sample_materials() for testing."
        )


def create_sample_materials(n_materials: int = 100) -> List[Dict]:
    """
    Create sample SAP-like material data with full multimodal information
    
    Args:
        n_materials: Number of materials to generate
        
    Returns:
        List of material dictionaries with all features
    """
    random.seed(42)
    np.random.seed(42)
    
    # Base data for generation
    material_types = {
        'STEEL': ['Steel', 'Carbon Steel', 'Alloy Steel'],
        'STAINLESS': ['Stainless Steel', 'Stainless'],
        'ALUMINUM': ['Aluminum', 'Aluminium'],
        'BRASS': ['Brass'],
        'PLASTIC': ['Plastic', 'Nylon', 'PVC'],
    }
    
    products = ['Bolt', 'Screw', 'Nut', 'Washer', 'Pin', 'Rivet']
    
    # Size specifications
    diameters = ['M6', 'M8', 'M10', 'M12', 'M16', 'M20']
    lengths = ['20', '25', '30', '40', '50', '60', '80', '100']
    
    # Standards
    standards = ['DIN 933', 'DIN 934', 'ISO 4017', 'ISO 4032', 'ANSI B18.2.1']
    
    # Coatings
    coatings = ['ZINC', 'CHROME', 'NICKEL', 'BLACK_OXIDE', 'NONE']
    
    # Thread types
    threads = ['METRIC', 'UNC', 'UNF']
    
    # Head types
    head_types = ['HEXAGONAL', 'SOCKET', 'FLAT', 'ROUND', 'COUNTERSUNK']
    
    # Strength classes (for bolts)
    strength_classes = ['4.6', '4.8', '5.8', '8.8', '10.9', '12.9']
    
    # Plants
    plants = [f'Plant_{i:04d}' for i in range(1001, 1021)]  # 20 plants
    
    # Suppliers
    suppliers = [f'SUPP_{i:06d}' for i in range(100000, 100150)]  # 150 suppliers
    
    # Material groups
    material_groups = {
        'Bolt': 'BOLTS',
        'Screw': 'SCREWS',
        'Nut': 'NUTS',
        'Washer': 'WASHERS',
        'Pin': 'PINS',
        'Rivet': 'RIVETS'
    }
    
    materials = []
    
    for i in range(n_materials):
        # Select product type
        product = random.choice(products)
        
        # Select material
        mat_category = random.choice(list(material_types.keys()))
        mat_name = random.choice(material_types[mat_category])
        
        thread = random.choice(threads)
        # Generate based on product type
        if product in ['Bolt', 'Screw']:
            diameter = random.choice(diameters)
            length = random.choice(lengths)
            standard = random.choice(standards)
            coating = random.choice(coatings)
            # thread = random.choice(threads)
            head_type = random.choice(head_types)
            strength = random.choice(strength_classes)
            
            # Description
            maktx = f"{mat_name} {product} {diameter}x{length} {standard}"
            
            # Characteristics
            characteristics = {
                'DIAMETER': diameter,
                'LENGTH': f"{length}mm",
                'MATERIAL': mat_category,
                'COATING': coating,
                'STANDARD': standard,
                'THREAD': thread,
                'HEAD_TYPE': head_type,
                'STRENGTH_CLASS': strength
            }
            
        elif product in ['Nut']:
            diameter = random.choice(diameters)
            standard = random.choice(standards)
            
            maktx = f"{mat_name} {product} {diameter} {standard}"
            
            characteristics = {
                'DIAMETER': diameter,
                'MATERIAL': mat_category,
                'STANDARD': standard,
                'THREAD': thread
            }
            
        elif product in ['Washer']:
            diameter = random.choice(diameters)
            
            maktx = f"{mat_name} {product} {diameter}"
            
            characteristics = {
                'DIAMETER': diameter,
                'MATERIAL': mat_category,
            }
            
        else:  # Pin, Rivet
            diameter = random.choice(['3', '4', '5', '6', '8'])
            length = random.choice(['10', '15', '20', '25', '30'])
            
            maktx = f"{mat_name} {product} {diameter}x{length}"
            
            characteristics = {
                'DIAMETER': f"{diameter}mm",
                'LENGTH': f"{length}mm",
                'MATERIAL': mat_category,
            }
        
        # Generate relational data
        # Number of plants (1-5)
        n_plants = random.randint(1, 5)
        mat_plants = random.sample(plants, n_plants)
        
        # Number of suppliers (1-3)
        n_suppliers = random.randint(1, 3)
        mat_suppliers = random.sample(suppliers, n_suppliers)
        
        # Usage patterns
        usage_frequency = random.randint(1, 100)
        po_count = random.randint(5, 200)
        last_po_days = random.randint(1, 365)
        
        # Price (correlated with material and size)
        base_price = {
            'STEEL': 0.20,
            'STAINLESS': 0.50,
            'ALUMINUM': 0.35,
            'BRASS': 0.60,
            'PLASTIC': 0.10
        }[mat_category]
        
        # Size factor
        size_num = int(diameter[1:]) if diameter.startswith('M') else int(diameter)
        size_factor = size_num / 8.0  # Normalize around M8
        
        price = base_price * size_factor * random.uniform(0.8, 1.2)
        
        # Build material dictionary
        material = {
            'MATNR': f'MAT{i+1:06d}',
            'MAKTX': maktx,
            'MATKL': material_groups[product],
            'MTART': 'FERT' if i % 3 == 0 else 'HALB',
            'MEINS': 'PC',
            'characteristics': characteristics,
            'plants': mat_plants,
            'suppliers': mat_suppliers,
            'usage_frequency': usage_frequency,
            'po_count': po_count,
            'last_po_days': last_po_days,
            'price_avg': round(price, 2),
            'plant_supplier_overlap': n_plants * n_suppliers  # Simplified
        }
        
        materials.append(material)
    
    return materials


def create_duplicate_pairs(
    materials: List[Dict],
    n_duplicates: int = 10
) -> List[Dict]:
    """
    Create intentional duplicate pairs for testing
    
    Args:
        materials: List of materials
        n_duplicates: Number of duplicate pairs to create
        
    Returns:
        List of materials with duplicates
    """
    duplicates = []
    
    for i in range(min(n_duplicates, len(materials))):
        original = materials[i].copy()
        
        # Create variant with slight modifications
        duplicate = original.copy()
        duplicate['MATNR'] = f"MAT{len(materials) + i + 1:06d}"
        
        # Modify description slightly
        desc = duplicate['MAKTX']
        variations = [
            desc.replace('Steel', 'Steel Alloy'),
            desc.replace('Bolt', 'Hex Bolt'),
            desc.replace('DIN', 'DIN-ISO'),
            desc + ' Grade A',
        ]
        duplicate['MAKTX'] = random.choice(variations)
        
        # Keep most characteristics the same
        duplicate['characteristics'] = original['characteristics'].copy()
        
        # Share some plants/suppliers
        duplicate['plants'] = original['plants'][:2]  # Partial overlap
        duplicate['suppliers'] = original['suppliers'][:1]  # Partial overlap
        
        duplicates.append(duplicate)
    
    return materials + duplicates


def materials_to_dataframe(materials: List[Dict]) -> pd.DataFrame:
    """
    Convert materials list to DataFrame (for compatibility)
    
    Args:
        materials: List of material dictionaries
        
    Returns:
        DataFrame with basic columns
    """
    df_data = []
    
    for mat in materials:
        row = {
            'MATNR': mat['MATNR'],
            'MAKTX': mat['MAKTX'],
            'MATKL': mat['MATKL'],
            'MTART': mat['MTART'],
            'MEINS': mat['MEINS'],
        }
        df_data.append(row)
    
    return pd.DataFrame(df_data)


def print_material_summary(material: Dict):
    """
    Print a human-readable summary of a material
    """
    print("=" * 60)
    print(f"Material: {material['MATNR']}")
    print(f"Description: {material['MAKTX']}")
    print(f"Group: {material['MATKL']}")
    print(f"Type: {material['MTART']}")
    
    print("\nCharacteristics:")
    for key, value in material.get('characteristics', {}).items():
        print(f"  {key}: {value}")
    
    print(f"\nPlants ({len(material.get('plants', []))}): {', '.join(material.get('plants', [])[:3])}")
    print(f"Suppliers ({len(material.get('suppliers', []))}): {', '.join(material.get('suppliers', [])[:3])}")
    print(f"Usage frequency: {material.get('usage_frequency', 0)}")
    print(f"Average price: €{material.get('price_avg', 0):.2f}")
    print("=" * 60)