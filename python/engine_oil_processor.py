import pandas as pd
import numpy as np
import re
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional, Set
import argparse


class EngineOilProcessor:
    def __init__(self):
        self.processed_data = {
            'sheet1': [],
            'sheet2': [],
            'standart': [],
            'premium': [],
            'oem': []
        }
        
        self.category_products = {
            'standart': "",
            'premium': "",
            'oem': ""
        }
        
        self.original_sheet1 = []
        self.original_sheet2 = []
        self.dynamic_base_oils = []
        
        # Product patterns for categorization
        self.product_patterns = {
            'oem': ["top quality", "premium - top quality", "premium-top quality"],
            'premium': [
                "premium", "premium - standart", "premium-standart",
                "standart - premium", "standart-premium", "standart alternativ",
                "full premium", "polimerli versiya / premium",
                "polimerli versiya/premium", "top premium"
            ],
            'standart': [
                "standart", "standard", "orjinal", "ekonomik", "ekonomik alternativ",
                "ekonomik alternativ (katık yok)", "katik yok", "ekonomik - standart",
                "ekonomik-standart", "ekonomik - standart - premium",
                "standart alternativ", "ekonomik-standart-premium",
                "standart - ekonomik", "standart-ekonomik",
                "polimerli versiya / standart", "polimerli versiya/standart",
                "polimerli standart", "çok ekonomik (müşteri onayı olmadan verilmemeli)",
                "çok ekonomik", "müşteri onayı olmadan verilmemeli"
            ]
        }
        
        # Size specifications
        self.liter_sizes = ["0.25L", "0.5L", "1L", "1.5L", "4L", "5L", "6L", "7L", "10L", "18L", "20L", "25L", "30L", "60L", "200L", "1000L"]
        self.kilo_sizes = ["4kg", "9kg", "14kg", "18kg", "180kg"]
        self.all_product_sizes = self.liter_sizes + self.kilo_sizes
        
        # Material specifications
        self.packaging_materials = ["Bidon", "Qapaq", "Etiket", "Qutu", "Palet"]
        self.all_materials = self.packaging_materials.copy()
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for pattern matching"""
        if not text:
            return ""
        
        # Convert to lowercase and handle Turkish characters
        text = str(text).lower()
        replacements = {
            'ğ': 'g', 'ü': 'u', 'ş': 's', 'ı': 'i',
            'ö': 'o', 'ç': 'c', 'ə': 'e'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove special characters and normalize spaces
        text = re.sub(r'[^\w\s-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_dynamic_base_oils(self, sheet1_data: List[List]) -> List[str]:
        """Extract dynamic base oils from sheet1 data"""
        base_oils = set()
        
        for i in range(1, len(sheet1_data)):
            row = sheet1_data[i]
            if not row or len(row) < 3:
                continue
                
            code = row[2] if len(row) > 2 else None
            if code and isinstance(code, str):
                trimmed_code = code.strip()
                # Check if it's a potential base oil code (2-3 uppercase letters)
                if (2 <= len(trimmed_code) <= 3 and 
                    trimmed_code.isupper() and 
                    trimmed_code.isalpha() and
                    trimmed_code not in self.packaging_materials):
                    base_oils.add(trimmed_code)
        
        return sorted(list(base_oils))
    
    def read_excel_file(self, filepath: str) -> Tuple[List[List], List[List]]:
        """Read Excel file and return sheet1 and sheet2 data as lists"""
        try:
            # Read both sheets
            df1 = pd.read_excel(filepath, sheet_name=0, header=None)
            df2 = pd.read_excel(filepath, sheet_name=1, header=None)
            
            # Convert to list of lists, handling NaN values
            sheet1_data = []
            for _, row in df1.iterrows():
                row_data = []
                for cell in row:
                    if pd.isna(cell):
                        row_data.append(None)
                    else:
                        row_data.append(cell)
                sheet1_data.append(row_data)
            
            sheet2_data = []
            for _, row in df2.iterrows():
                row_data = []
                for cell in row:
                    if pd.isna(cell):
                        row_data.append(None)
                    else:
                        row_data.append(cell)
                sheet2_data.append(row_data)
            
            return sheet1_data, sheet2_data
            
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return [], []
    
    def get_material_characteristic(self, material: str, size: str, product_name: str) -> str:
        """Get material characteristic based on material type, size, and product"""
        formatted_size = size.replace("L", " L") if size.endswith("L") else size.replace("kg", " kg")
        
        characteristics = {
            "Bidon": f"{formatted_size} Açıq Göy Aminol (öz istehsal)",
            "Qapaq": f"Qırmızı {formatted_size} (50/22 telescopic)",
            "Etiket": f"{product_name} (BK-06)",
            "Qutu": f"Kıpaj Aminol {formatted_size}",
            "Palet": "1.14x1.14"
        }
        
        # For dynamic base oils, return default characteristic
        if material in self.dynamic_base_oils:
            return "<Xarakteristika yoxdur>"
        
        return characteristics.get(material, f"{material} {formatted_size}")
    
    def generate_single_row_category_data(self, sheet1_row: List, sheet2_row: List, 
                                        category: str, product_name: str, 
                                        matched_pattern: str, additives_data: Dict) -> List:
        """Generate single row data for category"""
        product_id = sheet2_row[0] if sheet2_row[0] else "AL/EO-0001"
        engine_oil = "Engine oil"
        pcmo = "Passenger car motor oil (PCMO)"
        spec_type = matched_pattern
        
        single_row = [
            product_id,
            engine_oil,
            pcmo,
            product_id,
            product_name,
            product_name,
            spec_type
        ]
        
        # Add material data for each size and material combination
        for size in self.all_product_sizes:
            for material in self.all_materials:
                single_row.append(material)
                
                is_additive = material in self.dynamic_base_oils
                
                if is_additive:
                    percentage = additives_data.get(material)
                    single_row.append("<Xarakteristika yoxdur>")
                    single_row.append("faiz")
                    single_row.append(f"{percentage}%" if percentage else "-")
                else:
                    single_row.append(self.get_material_characteristic(material, size, product_name))
                    single_row.append("adəd")
                    single_row.append("1")
        
        return single_row
    
    def categorize_products(self, sheet1_data: List[List], sheet2_data: List[List]):
        """Categorize products based on patterns"""
        self.processed_data['standart'] = []
        self.processed_data['premium'] = []
        self.processed_data['oem'] = []
        
        for category in self.category_products:
            self.category_products[category] = ""
        
        i = 1
        while i < len(sheet1_data):
            header_row = sheet1_data[i]
            if not header_row or len(header_row) < 3:
                i += 1
                continue
                
            product_name_in_block = header_row[1]
            is_product_block_header = product_name_in_block and header_row[2]
            
            if is_product_block_header:
                block_text = ""
                block_end_index = i
                additives_data = {}
                
                # Process the block
                for j in range(i, len(sheet1_data)):
                    current_row = sheet1_data[j]
                    if not current_row:
                        continue
                    
                    # Check if we've reached the next product block
                    if j > i and len(current_row) > 2 and current_row[1] and current_row[2]:
                        break
                    
                    block_end_index = j
                    
                    if current_row[1]:
                        block_text += " " + str(current_row[1])
                    
                    # Extract additive data
                    additive_code = current_row[2] if len(current_row) > 2 else None
                    additive_percent = current_row[3] if len(current_row) > 3 else None
                    
                    if (additive_code and additive_percent and 
                        str(additive_code).strip() in self.all_materials):
                        additives_data[str(additive_code).strip()] = additive_percent
                
                # Normalize block text for pattern matching
                normalized_block_text = self.normalize_text(block_text)
                found_matches = {}
                
                # Check for pattern matches
                for category, patterns in self.product_patterns.items():
                    for pattern in patterns:
                        if self.normalize_text(pattern) in normalized_block_text:
                            found_matches[category] = pattern
                            break
                
                if found_matches:
                    # Find matching product in sheet2
                    potential_matches = []
                    for k in range(1, len(sheet2_data)):
                        sheet2_row = sheet2_data[k]
                        if not sheet2_row or len(sheet2_row) < 2:
                            continue
                        
                        sheet2_product_name = sheet2_row[1]
                        if (sheet2_product_name and 
                            self.normalize_text(str(product_name_in_block)) in 
                            self.normalize_text(str(sheet2_product_name))):
                            potential_matches.append(sheet2_row)
                    
                    # Select best match
                    best_match_row = None
                    if len(potential_matches) == 1:
                        best_match_row = potential_matches[0]
                    elif len(potential_matches) > 1:
                        # Sort by length and take shortest
                        potential_matches.sort(key=lambda x: len(str(x[1]) if x[1] else ""))
                        best_match_row = potential_matches[0]
                    
                    if best_match_row:
                        sheet2_product_name = best_match_row[1]
                        for category, matched_pattern in found_matches.items():
                            category_data = self.generate_single_row_category_data(
                                header_row, best_match_row, category,
                                str(sheet2_product_name), matched_pattern, additives_data
                            )
                            self.processed_data[category].append(category_data)
                            
                            if not self.category_products[category]:
                                self.category_products[category] = str(sheet2_product_name)
                
                i = block_end_index + 1
            else:
                i += 1
    
    def create_category_headers(self) -> List[str]:
        """Create headers for category sheets"""
        headers = [
            "ID", "Engine oil", "PCMO", "Məhsulun ID", "Məhsulun adı",
            "Spesifikasıyanın adı", "Spesifikasıyanın tipi"
        ]
        
        for size in self.all_product_sizes:
            for material in self.all_materials:
                formatted_size = size.replace("L", " L") if size.endswith("L") else size.replace("kg", " kg")
                headers.extend([
                    f"Material ({formatted_size} - {material})",
                    f"Xarakteristika ({formatted_size} - {material})",
                    f"Ölçü vahidi ({formatted_size} - {material})",
                    f"Miqdar ({formatted_size} - {material})"
                ])
        
        return headers
    
    def process_file(self, input_file: str, output_file: str = None):
        """Process the Excel file"""
        if not Path(input_file).exists():
            print(f"Error: File {input_file} not found!")
            return False
        
        print(f"Processing file: {input_file}")
        
        # Read Excel file
        sheet1_data, sheet2_data = self.read_excel_file(input_file)
        
        if not sheet1_data or not sheet2_data:
            print("Error: Could not read Excel sheets or file doesn't have required sheets!")
            return False
        
        # Extract dynamic base oils and update materials
        self.dynamic_base_oils = self.extract_dynamic_base_oils(sheet1_data)
        self.all_materials = self.packaging_materials + self.dynamic_base_oils
        
        print(f"Found dynamic base oils: {self.dynamic_base_oils}")
        
        # Store original data
        self.original_sheet1 = sheet1_data
        self.original_sheet2 = sheet2_data
        self.processed_data['sheet1'] = sheet1_data
        self.processed_data['sheet2'] = sheet2_data
        
        # Categorize products
        self.categorize_products(sheet1_data, sheet2_data)
        
        # Print categorization results
        print("\nCategorization Results:")
        for category in ['standart', 'premium', 'oem']:
            count = len(self.processed_data[category])
            product = self.category_products[category] or "None"
            print(f"{category.capitalize()}: {count} products - Main product: {product}")
        
        # Save to Excel
        if output_file is None:
            output_file = "Processed_Engine_Oil_Data.xlsx"
        
        self.save_to_excel(output_file)
        print(f"\nProcessed data saved to: {output_file}")
        
        return True
    
    def save_to_excel(self, output_file: str):
        """Save processed data to Excel file"""
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Save original sheets
            if self.processed_data['sheet1']:
                df1 = pd.DataFrame(self.processed_data['sheet1'])
                df1.to_excel(writer, sheet_name='Sheet1', index=False, header=False)
            
            if self.processed_data['sheet2']:
                df2 = pd.DataFrame(self.processed_data['sheet2'])
                df2.to_excel(writer, sheet_name='Sheet2', index=False, header=False)
            
            # Save category sheets with headers
            headers = self.create_category_headers()
            
            for category in ['standart', 'premium', 'oem']:
                if self.processed_data[category]:
                    # Create DataFrame with headers
                    data_with_headers = [headers] + self.processed_data[category]
                    df = pd.DataFrame(data_with_headers)
                    df.to_excel(writer, sheet_name=category.capitalize(), index=False, header=False)
    
    def display_summary(self):
        """Display processing summary"""
        print("\n" + "="*50)
        print("PROCESSING SUMMARY")
        print("="*50)
        
        print(f"Original Sheet1 rows: {len(self.original_sheet1)}")
        print(f"Original Sheet2 rows: {len(self.original_sheet2)}")
        print(f"Dynamic base oils found: {len(self.dynamic_base_oils)}")
        print(f"Total materials: {len(self.all_materials)}")
        
        print("\nCategorized Products:")
        for category in ['standart', 'premium', 'oem']:
            count = len(self.processed_data[category])
            product = self.category_products[category] or "None"
            print(f"  {category.capitalize()}: {count} products")
            if product != "None":
                print(f"    Main product: {product}")
        
        print("\nDynamic Base Oils:")
        for oil in self.dynamic_base_oils:
            print(f"  - {oil}")


def main():
    parser = argparse.ArgumentParser(
        description="Engine Oil Excel Data Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python engine_oil_processor.py all_products.xlsx
  python engine_oil_processor.py all_products.xlsx -o processed_data.xlsx
  python engine_oil_processor.py all_products.xlsx --summary
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Input Excel file path (all_products.xlsx)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='Processed_Engine_Oil_Data.xlsx',
        help='Output Excel file path (default: Processed_Engine_Oil_Data.xlsx)'
    )
    
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Display detailed processing summary'
    )
    
    args = parser.parse_args()
    
    # Create processor instance
    processor = EngineOilProcessor()
    
    # Process the file
    success = processor.process_file(args.input_file, args.output)
    
    if success:
        if args.summary:
            processor.display_summary()
        print("\nProcessing completed successfully!")
    else:
        print("Processing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()