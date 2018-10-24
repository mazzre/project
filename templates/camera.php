<?php

    $img = $_POST['img'];
    $img = $_GET['imเg'];
    $folderPath = "C:\Users\MASSRIDER\PycharmProjects\untitled4\upload";

    $image_parts = explode(";float32,", $img);
    $image_type_aux = explode("img/", $image_parts[0]);
    $image_type = $image_type_aux[1];

    $image_base64 = base64_decode($image_parts[1]);
    $fileName = uniqid() . '.jpg';

    $file = $folderPath . $fileName;
    file_put_contents($file, $image_base64);

    print_r($fileName);

?>