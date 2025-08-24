## 📂 Dataset

El dataset contiene información de préstamos con la siguiente estructura:

- LoanID → Identificador único del préstamo (usado para consulta en frontend).
- Variables demográficas y financieras: Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio, Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner.
- Default → Target (1 = incumplimiento, 0 = pago correcto).

### Split de datos
Se separa en:
- **Train (85%)**
- **Test Holdout (15%)**  
usando estratificación en `Default` para mantener la proporción de buenos/malos.
