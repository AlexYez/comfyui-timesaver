/**
 * 3D-превью ракурса для TS Angle Select.
 *
 * Объект, вокруг него орбита, на орбите камера, от камеры к объекту — рельс
 * приближения с тремя засечками.
 *
 * ⚠️ Сцена НИЧЕГО не принимает от мыши. Настраивать три величины одним
 * движением по одному холсту оказалось неудобно (замечание владельца пака):
 * поворот, высота и крупность живут на трёх отдельных регуляторах в
 * `ts-angle-select.js`, а сцена только показывает, что получилось.
 *
 * ⚠️ Three.js подтягивается ЛЕНИВО и НЕ из `js/`: ComfyUI импортирует каждый
 * `.js` из веб-папки пака при загрузке страницы, и 675 КБ платили бы все, даже
 * никогда не поставив эту ноду. Библиотека лежит в `nodes/text/_vendor/` и
 * отдаётся маршрутом `/ts_angle_select/three.module.js`.
 *
 * ⚠️ Модуль ничего не знает ни про ноду, ни про промпты: наружу он отдаёт
 * только `{azimuth, height, framing}` и принимает то же самое. Всё остальное —
 * забота `ts-angle-select.js`.
 */

const THREE_URL = "/ts_angle_select/three.module.js";

/** Высоты камеры в градусах — те же четыре, что понимает бэкенд. */
export const HEIGHT_ANGLES = {
    low: -22,
    "eye-level": 4,
    elevated: 26,
    high: 52,
};

/** Крупность — это расстояние. Числа подобраны так, чтобы разница читалась. */
export const FRAMING_DISTANCE = {
    wide: 4.6,
    medium: 3.1,
    "close-up": 1.9,
};


let threePromise = null;

/** Один запрос на страницу, сколько бы нод ни стояло в графе. */
export function loadThree() {
    if (!threePromise) {
        threePromise = import(/* webpackIgnore: true */ THREE_URL).catch((error) => {
            threePromise = null;
            throw error;
        });
    }
    return threePromise;
}

/**
 * Собрать сцену в контейнере.
 *
 * @param {object} options
 * @param {HTMLElement} options.container куда рисовать
 * @param {object} options.THREE       уже загруженный модуль
 * @param {object} options.colors      палитра из общей темы
 * @param {object} options.state       {azimuth, height, framing}
 */
export function createAngleScene({ container, THREE, colors, state }) {
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(2, globalThis.devicePixelRatio || 1));
    renderer.domElement.style.cssText = "position:absolute;inset:0;width:100%;height:100%;display:block";
    const element = renderer.domElement;
    container.appendChild(element);

    const scene = new THREE.Scene();
    const view = new THREE.PerspectiveCamera(34, 1, 0.1, 100);
    // Зритель смотрит на площадку сбоку и чуть сверху: с этой точки видно и
    // орбиту, и высоту камеры, и расстояние до объекта одновременно. Точную
    // дистанцию подбирает `frameScene` — она зависит от пропорций ноды.
    const VIEW_TARGET = new THREE.Vector3(0, 1.0, 0);
    // ⚠️ Взгляд заметно сверху. На пологом ракурсе камера в ближних
    // положениях (45° и 225°) налезала на фигуру, и оба предмета сливались
    // в одно пятно — то есть сцена переставала отвечать на свой вопрос.
    const VIEW_DIR = new THREE.Vector3(0.52, 0.78, 0.56).normalize();

    function frameScene() {
        // ⚠️ Отодвигаемся ровно настолько, чтобы орбита влезла и по ширине, и
        // по высоте: на узкой ноде ограничение даёт ширина, на широкой — высота.
        // ⚠️ Помещаться обязана не орбита, а вся установка целиком: на общем
        // плане камера отъезжает ЗА орбиту, и на прежнем кадрировании она
        // уходила за нижний край.
        const radius = 6.4;
        const vertical = radius / Math.tan((view.fov * Math.PI) / 360);
        const horizontal = vertical / Math.max(0.35, view.aspect);
        const distance = Math.max(vertical, horizontal) * 0.8;
        view.position.copy(VIEW_TARGET).addScaledVector(VIEW_DIR, distance);
        view.lookAt(VIEW_TARGET);
    }

    scene.add(new THREE.AmbientLight(0xffffff, 1.5));
    const key = new THREE.DirectionalLight(0xffffff, 1.4);
    key.position.set(4, 7, 5);
    scene.add(key);

    const line = (hex, opacity = 1) =>
        new THREE.LineBasicMaterial({ color: hex, transparent: true, opacity });
    const solid = (hex, opacity = 1) =>
        new THREE.MeshStandardMaterial({
            color: hex, roughness: 0.62, metalness: 0.05,
            transparent: opacity < 1, opacity,
        });

    const palette = {
        accent: new THREE.Color(colors.accent),
        border: new THREE.Color(colors.border),
        muted: new THREE.Color(colors.muted),
        text: new THREE.Color(colors.text),
    };

    // ---- площадка --------------------------------------------------------
    const grid = new THREE.GridHelper(11, 22, palette.border, palette.border);
    grid.material.transparent = true;
    grid.material.opacity = 0.22;
    scene.add(grid);

    const ringGeometry = new THREE.RingGeometry(3.02, 3.06, 96);
    const ring = new THREE.Mesh(
        ringGeometry,
        new THREE.MeshBasicMaterial({
            color: palette.muted, side: THREE.DoubleSide, transparent: true, opacity: 0.85,
        }),
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.01;
    scene.add(ring);

    // Восемь засечек на орбите — куда камера встаёт.
    const detents = new THREE.Group();
    for (let angle = 0; angle < 360; angle += 45) {
        const dot = new THREE.Mesh(
            new THREE.CircleGeometry(0.11, 20),
            new THREE.MeshBasicMaterial({ color: palette.muted, transparent: true, opacity: 0.9 }),
        );
        const rad = (angle * Math.PI) / 180;
        dot.position.set(Math.sin(rad) * 3.04, 0.02, Math.cos(rad) * 3.04);
        dot.rotation.x = -Math.PI / 2;
        detents.add(dot);
    }
    scene.add(detents);

    // ---- объект ----------------------------------------------------------
    // Условная фигура из примитивов: нужна не похожесть, а понятные «перёд»,
    // «верх» и масштаб, относительно которых читается ракурс.
    const subject = new THREE.Group();
    const skin = solid(palette.muted);
    const shade = solid(palette.muted.clone().multiplyScalar(0.78));
    const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.28, 0.6, 6, 18), skin);
    torso.position.y = 1.02;
    torso.scale.z = 0.7;
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.22, 24, 18), skin);
    head.position.y = 1.66;
    const hips = new THREE.Mesh(new THREE.CapsuleGeometry(0.2, 0.46, 6, 14), shade);
    hips.position.y = 0.4;
    hips.scale.z = 0.78;
    // ⚠️ Нос и плечи — единственные детали, которые говорят, куда фигура
    // смотрит. Без них «спереди» и «сзади» на площадке неотличимы, а ракурс
    // ровно об этом.
    const nose = new THREE.Mesh(new THREE.ConeGeometry(0.075, 0.22, 12), solid(palette.accent));
    nose.position.set(0, 1.64, 0.24);
    nose.rotation.x = Math.PI / 2;
    const shoulders = new THREE.Mesh(new THREE.BoxGeometry(0.78, 0.11, 0.24), shade);
    shoulders.position.y = 1.33;
    subject.add(torso, head, hips, nose, shoulders);
    scene.add(subject);

    // ---- камера-гизмо ----------------------------------------------------
    const gizmo = new THREE.Group();
    const cameraMaterial = solid(palette.accent);
    const bodyMesh = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.42, 0.42), cameraMaterial);
    const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.17, 0.22, 0.34, 20), cameraMaterial);
    lens.rotation.x = Math.PI / 2;
    lens.position.z = 0.36;
    const hood = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.14, 0.22), cameraMaterial);
    hood.position.set(-0.03, 0.27, -0.03);
    // Тонкая обводка по корпусу — камера читается силуэтом даже на тёмном фоне.
    const outline = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(0.6, 0.42, 0.42)),
        line(palette.text, 0.45),
    );
    gizmo.add(bodyMesh, lens, hood, outline);
    scene.add(gizmo);

    // ⚠️ Конус зрения — КАРКАС, а не залитый объём: сплошной конус закрывал
    // собой и объект, и половину площадки, то есть мешал ровно тому, ради чего
    // сцену и рисуют.
    // ⚠️ Четыре луча и рамка, а не каркас конуса: каркас давал сетку линий
    // поперёк фигуры — красиво в вакууме и нечитаемо в ноде.
    const frustumGeometry = new THREE.BufferGeometry();
    frustumGeometry.setAttribute(
        // ⚠️ 16 вершин, а не 12: четыре луча (по две точки) плюс четыре стороны
        // рамки (тоже по две) — ровно восемь отрезков. В буфере на 12 вершин
        // последние стороны не помещались, и рамка кадра выглядела разорванной.
        "position", new THREE.BufferAttribute(new Float32Array(16 * 3), 3));
    const frustum = new THREE.LineSegments(frustumGeometry, line(palette.accent, 0.5));
    scene.add(frustum);

    // ---- линия расстояния ------------------------------------------------
    // ⚠️ Просто отрезок от объекта до камеры. Раньше на нём сидели три засечки
    // и ручка — ими двигали камеру; теперь приближение живёт на отдельном
    // регуляторе, а засечки оставались висеть ЗА камерой на ближнем плане и
    // читались как посторонний мусор в сцене.
    const railGeometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(), new THREE.Vector3(),
    ]);
    const rail = new THREE.Line(railGeometry, line(palette.accent, 0.8));
    scene.add(rail);

    // ---- состояние -------------------------------------------------------
    const current = {
        azimuth: state.azimuth,
        height: state.height,
        framing: state.framing,
    };
    // Непрерывные величины, по которым идёт перетаскивание; к детентам они
    // притягиваются при каждом кадре.
    const smooth = {
        azimuth: current.azimuth,
        elevation: HEIGHT_ANGLES[current.height] ?? 4,
        distance: FRAMING_DISTANCE[current.framing] ?? 3.1,
    };

    function cameraPosition() {
        const az = (smooth.azimuth * Math.PI) / 180;
        const el = (smooth.elevation * Math.PI) / 180;
        const r = smooth.distance;
        return new THREE.Vector3(
            Math.sin(az) * Math.cos(el) * r,
            1.15 + Math.sin(el) * r,
            Math.cos(az) * Math.cos(el) * r,
        );
    }

    const target = new THREE.Vector3(0, 1.15, 0);

    frameScene();

    function layout() {
        const position = cameraPosition();
        gizmo.position.copy(position);
        gizmo.lookAt(target);

        const toTarget = new THREE.Vector3().subVectors(target, position);
        const length = toTarget.length();
        const forward = toTarget.clone().normalize();
        // Рамка кадра — прямо на объекте, лучи — от объектива к её углам.
        const right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0, 1, 0))
            .normalize().multiplyScalar(length * 0.3);
        const up = new THREE.Vector3().crossVectors(right, forward)
            .normalize().multiplyScalar(length * 0.22);
        const corners = [
            target.clone().add(right).add(up),
            target.clone().sub(right).add(up),
            target.clone().sub(right).sub(up),
            target.clone().add(right).sub(up),
        ];
        const points = [];
        for (const corner of corners) points.push(position, corner);
        for (let i = 0; i < 4; i += 1) points.push(corners[i], corners[(i + 1) % 4]);
        const array = frustum.geometry.attributes.position.array;
        points.forEach((point, index) => {
            array[index * 3] = point.x;
            array[index * 3 + 1] = point.y;
            array[index * 3 + 2] = point.z;
        });
        frustum.geometry.attributes.position.needsUpdate = true;
        frustum.geometry.computeBoundingSphere();

        railGeometry.setFromPoints([target.clone(), position.clone()]);
        detents.children.forEach((dot, index) => {
            const active = index * 45 === current.azimuth;
            dot.material.color.copy(active ? palette.accent : palette.muted);
            dot.material.opacity = active ? 1 : 0.75;
            dot.scale.setScalar(active ? 1.5 : 1);
        });
    }

    function render() {
        layout();
        renderer.render(scene, view);
    }

    function resize() {
        const width = Math.max(1, container.clientWidth);
        const height = Math.max(1, container.clientHeight);
        renderer.setSize(width, height, false);
        view.aspect = width / height;
        view.updateProjectionMatrix();
        frameScene();
        render();
    }

    return {
        element,
        resize,
        render,
        setState(next) {
            current.azimuth = next.azimuth;
            current.height = next.height;
            current.framing = next.framing;
            smooth.azimuth = current.azimuth;
            smooth.elevation = HEIGHT_ANGLES[current.height] ?? 4;
            smooth.distance = FRAMING_DISTANCE[current.framing] ?? 3.1;
            render();
        },
        dispose() {
            renderer.dispose();
            scene.traverse((object) => {
                object.geometry?.dispose?.();
                if (Array.isArray(object.material)) object.material.forEach((m) => m.dispose?.());
                else object.material?.dispose?.();
            });
            element.remove();
        },
    };
}
