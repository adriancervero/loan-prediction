
# Data paths

DATA_RAW = "../data/raw"
DATA_INTERIM = "../data/interim/loans_interim.csv"
TRAIN_PATH = "../data/processed/train.csv"
TEST_PATH = "../data/processed/test.csv"

# Prepare data variables
CHUNKSIZE = 10 ** 6
YEAR = '2017'

HEADERS = [
    "Pool_ID",
    "Loan_ID",
    "Month_Reporting_Period",
    "Channel",
    "Seller_Name",
    "Servicer_Name",
    "Master_Servicer",
    "Original_Interest_Rate",
    "Current_Interest_Rate",
    "Original_UPB",
    "UPB_at_Issuance",
    "Current_Actual_UPB",
    "Original_Loan_Term",
    "Origination_Date",
    "First_Payment_Date",
    "Loan_Age",
    "Months_to_Legal_Maturity",
    "Months_to_Maturity",
    "Maturity_Date",
    "LTV",
    "CLTV",
    "Number_of_Borrowers",
    "DTI",
    "Borrower_Credit_Score",
    "Co-Borrower_Credit_Score",
    "First_Time_Home_Buyer",
    "Loan_Purpose",
    "Property_Type",
    "Number_of_Units",
    "Occupancy_Status",
    "Property_State",
    "MSA",
    "Zip",
    "Mortgage_Insurance_Per",
    "Amortization_Type",
    "Prepayment_Penalty_Indicator",
    "Interest_Only_Loan_Indicator",
    "First_Principal_Interest_Payment_date",
    "Months_to_Amortization",
    "Current_Loan_Delinquency_Status",
    "Loan_Payment_History",
    "Modification_Flag",
    "Mortgage_Insurance_Cancellation",
    "Zero_Balance_Code",
    "Zero_Balance_Effective_Date",
    "UPB_at_Time_Removal",
    "Repurchase_Date",
    "Scheduled_Principal_Current",
    "Total_Principal_Current",
    "Unscheduled_Principal_Current",
    "Last_Paid_Installment_Date",
    "Foreclosure_Date",
    "Disposition_Date",
    "Foreclosure_Costs",
    "Property_Preservation_And_Repair_Costs",
    "Asset_Recovery_Costs",
    "Miscellaneous_Holding_Expenses",
    "Associated_Taxes",
    "Net_Sales_Proceeds",
    "Credit_Enhancement_Proceeds",
    "Repurchase_Make_Whole_Proceeds",
    "Other_Foreclosure_Proceeds",
    "UPB",
    "Principal_Forgiveness_Amount",
    "Original_List_Start_Date",
    "Original_List_Price",
    "Current_List_Start_Date",
    "Current_List_Price",
    "Borrower_Credit_Score_At_Issuance",
    "Co-Borrower_Credit_Score_At_Issuance",
    "Borrower_Credit_Score_Current",
    "Co-Borrower_Credit_Score_Current",
    "Mortgage_Insurance_Type",
    "Servicing_Activity_Indicator",
    "Current_Period_Modification_Loss_Amount",
    "Cumulative_Modification_Loss_Amount",
    "Current_Period_Credit_Event",
    "Cumulative_Credit_Event",
    "HomeReady_Indicator",
    "Foreclosure_Principal_Write-off",
    "Relocation_Mortgage",
    "Zero_Balance_Code_Change_Date",
    "Loan_Holdback",
    "Loan_Holdback_Effective_Date",
    "Delinquent_Accrued",
    "Property_Valuation_Method",
    "High_Balance_Loan",
    "ARM_Period_<=5",
    "ARM_Product_Type",
    "Initial_Fixed-Rate_Period",
    "Interest_Rate_Adjustament_Frequency",
    "Next_Interest_Rate_Adjustment_Date",
    "Next_Payment_Change_Date",
    "Index",
    "ARM_Cap_Structure",
    "Initial_Interest_Rate_Cap_Up",
    "Periodic_Interest_Rate_Cap_Up",
    "Lifetime_Interest_Rate",
    "Mortgage_Margin",
    "ARM_Ballon",
    "ARM_Plan_Number",
    "Borrower_Assistance_Plan",
    "High_Loan_to_Value",
    "Deal_Name",
    "Repurchase_Make_Whole_Proceeds_Flag",
    "Delinquency_Resolution",
    "Delinquency_Resolution_Count",
    "Total_Deferral_Amount",
]

SELECT = [
    'Loan_ID',
    'Channel',
    'Seller_Name',
    'Original_Interest_Rate',
    'Original_UPB',
    'Original_Loan_Term',
    "Origination_Date",
    "First_Payment_Date",
    "LTV",
    "CLTV",
    "Number_of_Borrowers",
    "DTI",
    "Borrower_Credit_Score",
    "Co-Borrower_Credit_Score",
    "First_Time_Home_Buyer",
    "Loan_Purpose",
    "Property_Type",
    "Number_of_Units",
    "Occupancy_Status",
    "Property_State",
    "Zip",
    "Mortgage_Insurance_Per",
    
    "Foreclosure_Date",
]

# Feature Engineering

DROP_COLS = ['Mortgage_Insurance_Per', 'Co-Borrower_Credit_Score']
RANDOM_STATE = 42

# Training
#NUMERICAL = ['Original_Interest_Rate', 'Original_UPB', 'Original_Loan_Term', 'LTV', 'CLTV', 'Number_of_Borrowers',
            # 'DTI', 'Borrower_Credit_Score', 'Co-Borrower_Credit_Score', 'Number_of_Units', 'Zip', 'Mortgage_Insurance_Per']

#CATEGORICAL = ['Channel', 'First_Time_Home_Buyer', 'Loan_Purpose', 'Property_Type', 'Occupancy_Status']

NUMERICAL = ['Original_Interest_Rate','CLTV',
             'Borrower_Credit_Score', 'orig_month']

CATEGORICAL = ['First_Time_Home_Buyer', 'Loan_Purpose', 'Property_Type']

TARGET = 'foreclosure'