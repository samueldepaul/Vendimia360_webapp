import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import base64


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
    img_to_bytes(r"Group 1header (2) (1).jpg")
)
st.markdown(
    header_html, unsafe_allow_html=True,
)

st.title("Contacto")

st.markdown(
    f'<div style="background-color:#FFBFBF; padding: 10px 25px; border-radius: 5px;"><h4 style="color:#320014; font-size: 16px;">Aquí puedes encontrar información sobre nosotros, sobre futuras funcionalidades que se añadirán a la plataforma, y un formulario de contacto para que puedas transladarnos tus dudas o problemas técnicos.</h4></div>',
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

# Define los colores
primaryColor = '#7F0055'
textColor = '#363636'
backgroundColor = '#FFFFFF'
backgroundLightColor = '#F4F7FC'
black = "#000000"

# Define el estilo de la página
st.markdown(f"""
    <style>
        .reportview-container {{
            background-color: {backgroundColor};
            color: {textColor};
        }}
        .sidebar .sidebar-content {{
            background-color: {primaryColor};
            color: {textColor};
        }}
        .stButton {{
            background-color: {primaryColor};
            color: {textColor};
        }}
        .main {{
            background-color: {backgroundLightColor};
            padding: 2rem;
            min-height: 100vh;
        }}
        h1 {{
            color: {black};
            padding-bottom: 0.5rem;
        }}
    </style>
""", unsafe_allow_html=True)

# Define el contenido de la página
st.markdown("""<div class="main">
        <h2>¿Quiénes somos?</h2>
        <p>Los desrrolladores de esta plataforma somos Samuel de Paúl y Franco Rojas, integrantes del equipo Nearest Neighbors y participantes en el Cajamar UniverityHack Datathon 2023.
        Esta plataforma se desarrolla como producto final de un proyecto dedicado principalmente a la predicción de la producción de viñedos. El objetivo es poder acercar la increíble 
        utilidad del Machine Learning a personas sin conocimiento técnico en el área; en este caso, viticultores y gente que trabaja en el campo.
        </p>
        <hr>
        <h2>Futuras Funcionalidades</h2>
        <p>Pretendemos continuar desarrollando progresivamente esta plataforma, por lo que podemos anticipar una lista de futuras funcionalidades:</p>
        <ul>
            <li>Elaboración Automática Cuaderno de Campo: Facilitar a los Viticultores el proceso de elaboración del cuaderno de campo que tienen que entregar cada año a la Conselleria con la producción y otra multitud de datos.</li>
            <li>Registro de Productos Fitosanitarios: Permitir llevar un registro unificado de los productos fitosanitarios y abonos utilizados en campañas anteriores. Posibilidad de calcular cantidades y proporciones de productos para elaborar mezcla.</li>
            <li>Posibilidad de introducir valores faltantes en modo Manual.</li>
            <li>Soporte para más unidades métricas como cuarteradas o fanecas.</li>
            <li>Soporte para múltiples idiomas.</li>
        </ul>
        <hr>
        <h2>Contacto</h2>
        <p>En caso de que te haya surgido alguna duda o tengas problemas técnicos con la plataforma, ponte en contacto con nosotros rellenando el siguiente formulario:</p>
        <form>
            <div class="form-group">
                <label for="inputNombre">Nombre:</label>
                <input type="text" class="form-control" id="inputNombre" placeholder="Introduce tu nombre">
            </div>
            <div class="form-group">
                <label for="inputEmail">Correo electrónico:</label>
                <input type="email" class="form-control" id="inputEmail" placeholder="Introduce tu correo electrónico">
            </div>
            <div class="form-group">
                <label for="inputAsunto">Asunto:</label>
                <input type="text" class="form-control" id="inputAsunto" placeholder="Introduce el asunto">
            </div>
            <div class="form-group">
                <label for="inputMensaje">Mensaje:</label>
                <textarea class="form-control" id="inputMensaje" rows="5"></textarea>
            </div>
            <button type="submit" class="btn btn-primary">Enviar</button>
        </form>
        <hr>
    </div>
""", unsafe_allow_html=True)