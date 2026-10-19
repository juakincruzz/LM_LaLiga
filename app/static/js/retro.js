// Comportamiento del tema retro: atajos 1–4, reloj de la barra de menú e
// interruptor del efecto CRT. Sin dependencias.
(function () {
    "use strict";

    // --- Atajos de teclado -------------------------------------------------
    // Cada botón de acción lleva data-tecla="n"; la barra inferior lleva
    // data-dispara="n" y simplemente pulsa el botón correspondiente, así que
    // HTMX sigue siendo el único que lanza trabajos.
    function pulsar(tecla) {
        var boton = document.querySelector('[data-tecla="' + tecla + '"]');
        if (!boton || boton.disabled) return;
        boton.classList.add("pulsado");
        setTimeout(function () { boton.classList.remove("pulsado"); }, 120);
        boton.click();
    }

    document.addEventListener("keydown", function (e) {
        if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
        var t = e.target;
        if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
        if (/^[1-4]$/.test(e.key)) {
            e.preventDefault();
            pulsar(e.key);
        }
    });

    document.addEventListener("click", function (e) {
        var origen = e.target.closest("[data-dispara]");
        if (origen && !origen.disabled) pulsar(origen.getAttribute("data-dispara"));
    });

    // --- Reloj ---------------------------------------------------------------
    var reloj = document.getElementById("reloj");
    function tic() {
        var d = new Date();
        reloj.textContent = String(d.getHours()).padStart(2, "0") + ":" + String(d.getMinutes()).padStart(2, "0");
    }
    if (reloj) {
        tic();
        setInterval(tic, 10000);
    }

    // --- Efecto CRT ------------------------------------------------------------
    // La preferencia se guarda solo en este navegador; si el almacenamiento no
    // está disponible, el interruptor funciona igual durante la visita.
    var raiz = document.documentElement;
    var interruptor = document.getElementById("conmutar-crt");
    function pintar() {
        var activo = !raiz.classList.contains("sin-crt");
        interruptor.setAttribute("aria-pressed", String(activo));
        interruptor.textContent = activo ? "CRT: ON" : "CRT: OFF";
    }
    if (interruptor) {
        pintar();
        interruptor.addEventListener("click", function () {
            raiz.classList.toggle("sin-crt");
            try {
                localStorage.setItem("crt", raiz.classList.contains("sin-crt") ? "off" : "on");
            } catch (e) { /* almacenamiento bloqueado: no pasa nada */ }
            pintar();
        });
    }
})();
