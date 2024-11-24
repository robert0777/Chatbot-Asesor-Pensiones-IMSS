
class QuestionClassifier:
       def __init__(self):
        # Common Spanish greetings and pleasantries
        self.greetings = {
            # Basic greetings
            'hola',
            'buenos días',
            'buenas tardes',
            'buenas noches',
            'saludos',
            
            # Informal greetings
            'qué tal',
            'cómo estás',
            'como estas',
            'qué hace',
            'que hace',
            'qué onda',
            'que onda',
            
            # Formal greetings
            'cómo le va',
            'cómo está usted',
            'como esta usted',
            'mucho gusto',
            
            # Time-specific greetings
            'bonito día',
            'feliz día',
            'buen día',
            
            # Common variations
            'hey',
            'hi',
            'hello',
            
            # Introduction phrases
            'cómo te llamas',
            'me llamo',
            'mi nombre es',
            
            # Combined greetings
            'hola, qué tal',
            'hola, cómo estás',
            'hola, buenos días',
            'hola, qué hace',
            
            # Regional variations
            'quiubo',
            'qué hubo',
            'que hubo',
            'qué hay',
            'que hay'
        }
        
       
        

        # Keywords related to social security domain, organized by categories
        self.domain_keywords = {
            # Instituciones y Organismos
            'imss',
            'issste',
            'consar',
            'consami',
            'infonavit',
            'pensionissste',
            'instituto',
            'siefore',
            'afore',
            'afores',
            'aseguradora',
            'aseguradoras',
            
            # Tipos de Pensiones y Beneficios
            'pensión',
            'pension',
            'pensiones',
            'jubilación',
            'jubilacion',
            'jubilaciones',
            'cesantía',
            'cesantia',
            'vejez',
            'invalidez',
            'incapacidad',
            'viudez',
            'orfandad',
            'retiro',
            'muerte',
            'maternidad',
            'riesgo',
            'riesgos',
            
            # Términos Financieros
            'aportación',
            'aportacion',
            'aportaciones',
            'cuota',
            'cuotas',
            'cotización',
            'cotizacion',
            'cotizaciones',
            'saldo',
            'ahorro',
            'fondo',
            'fondos',
            'intereses',
            'inversión',
            'comisión',
            'comisiones',
            'subcuenta',
            'rendimiento',
            'rendimientos',
            
            # Términos Laborales
            'salario',
            'salarios',
            'sbc',  # Salario Base de Cotización
            'sueldo',
            'sueldos',
            'trabajador',
            'trabajadora',
            'trabajadores',
            'trabajadoras',
            'patrón',
            'patron',
            'patrones',
            'empresa',
            'empresas',
            'empleador',
            'empleadores',
            'obrero-patronal',
            
            # Conceptos Legales
            'ley',
            'leyes',
            'artículo',
            'articulo',
            'artículos',
            'articulos',
            'derecho',
            'derechos',
            'reglamento',
            'legislación',
            'legislacion',
            'disposición',
            'disposicion',
            'disposiciones',
            'jurídica',
            'juridica',
            'legal',
            'legales',
            
            # Beneficiarios y Derechohabientes
            'beneficiario',
            'beneficiarios',
            'derechohabiente',
            'derechohabientes',
            'asegurado',
            'asegurada',
            'asegurados',
            'aseguradas',
            'pensionado',
            'pensionada',
            'pensionados',
            'pensionadas',
            'viuda',
            'viudo',
            'huérfano',
            'huerfano',
            'huérfana',
            'huerfana',
            'esposa',
            'esposo',
            'concubina',
            'concubino',
            
            # Requisitos y Condiciones
            'semanas',
            'cotizadas',
            'edad',
            'antiguedad',
            'antigüedad',
            'requisito',
            'requisitos',
            'condiciones',
            'vigencia',
            'vigente',
            'vigentes',
            
            # Prestaciones y Servicios
            'prestación',
            'prestacion',
            'prestaciones',
            'servicio',
            'servicios',
            'subsidio',
            'subsidios',
            'ayuda',
            'ayudas',
            'beneficio',
            'beneficios',
            'asistencial',
            'asistenciales',
            
            # Cálculos y Montos
            'cálculo',
            'calculo',
            'monto',
            'montos',
            'importe',
            'cuantía',
            'cantidad',
            'cantidades',
            'porcentaje',
            'factor',
            'índice',
            'indice',
            
            # Procedimientos y Trámites
            'solicitud',
            'solicitar',
            'trámite',
            'tramite',
            'procedimiento',
            'formato',
            'formatos',
            'documentos',
            'constancia',
            'dictamen',
            
            # Modalidades y Regímenes
            'modalidad',
            'modalidades',
            'régimen',
            'regimen',
            'regímenes',
            'esquema',
            'esquemas',
            'plan',
            'planes',
            
            # Pagos y Periodos
            'pago',
            'pagos',
            'mensual',
            'anual',
            'bimestral',
            'periodo',
            'periodos',
            'fecha',
            'plazo',
            'plazos',
            
            # Riesgos y Contingencias
            'accidente',
            'accidentes',
            'enfermedad',
            'enfermedades',
            'contingencia',
            'contingencias',
            'siniestro',
            'fallecimiento',
            'defunción',
            'defuncion',
            
            # Términos Económicos
            'mínimo',
            'minimo',
            'máximo',
            'maximo',
            'incremento',
            'incrementos',
            'actualización',
            'actualizacion',
            'ajuste',
            'ajustes',
            
            # Otros Términos Relevantes
            'voluntario',
            'voluntaria',
            'obligatorio',
            'obligatoria',
            'permanente',
            'temporal',
            'parcial',
            'total',
            'garantizada',
            'vitalicia',
            'constitutivo',
            'solidario',
            'tripartita'
        }
        
        
