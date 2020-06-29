$(document).ready(function(){

    const generate_similar = img => () => {
        $.ajax({
            url     : "generate_similar",
            type    : 'POST',
            data    : JSON.stringify({"img"   :img,
                                      "text"  :$('#text')[0].value,
                                      "color1":$('#color_1')[0].value,
                                      "color2":$('#color_2')[0].value,
                                      "color3":$('#color_3')[0].value,
                                      "color4":$('#color_4')[0].value,
                                      "color5":$('#color_5')[0].value,
                                      "value1":$('#value_1')[0].value,
                                      "value2":$('#value_2')[0].value,
                                      "value3":$('#value_3')[0].value,
                                      "value4":$('#value_4')[0].value,
                                      "value5":$('#value_5')[0].value,
                                     }),
            success : function(response){
                $('#img_1').attr("src", response["img_path1"]);
                $('#img_2').attr("src", response["img_path2"]);
                $('#img_3').attr("src", response["img_path3"]);
                $('#img_4').attr("src", response["img_path4"]);
                $('#img_5').attr("src", response["img_path5"]);
                $('#img_6').attr("src", response["img_path6"]);
                $('#img_7').attr("src", response["img_path7"]);
                $('#img_8').attr("src", response["img_path8"]);
                $('#img_9').attr("src", response["img_path9"]);
            },
            error   : err => {alert("Something went wrong");},
            contentType : 'application/json',
            dataType    : 'json',
        });
    };


    const generate_random = img => () => {
        $.ajax({
            url     : "generate_random",
            type    : 'POST',
            data    : JSON.stringify({"text"  : $('#text')[0].value,
                                      "color1":$('#color_1')[0].value,
                                      "color2":$('#color_2')[0].value,
                                      "color3":$('#color_3')[0].value,
                                      "color4":$('#color_4')[0].value,
                                      "color5":$('#color_5')[0].value,
                                      "value1":$('#value_1')[0].value,
                                      "value2":$('#value_2')[0].value,
                                      "value3":$('#value_3')[0].value,
                                      "value4":$('#value_4')[0].value,
                                      "value5":$('#value_5')[0].value,
                                     }),
            success : function(response){
                $('#img_1').attr("src", response["img_path1"]);
                $('#img_2').attr("src", response["img_path2"]);
                $('#img_3').attr("src", response["img_path3"]);
                $('#img_4').attr("src", response["img_path4"]);
                $('#img_5').attr("src", response["img_path5"]);
                $('#img_6').attr("src", response["img_path6"]);
                $('#img_7').attr("src", response["img_path7"]);
                $('#img_8').attr("src", response["img_path8"]);
                $('#img_9').attr("src", response["img_path9"]);
            },
            error   : err => {alert("Something went wrong");},
            contentType : 'application/json',
            dataType    : 'json',
        });
    };

    // initialize on multiple elements with jQuery
    $('.color-input').each( function( i, elem ) {
        var hueb = new Huebee( elem, {
            customColors: ['rgb(0,0,0)', 'rgb(0,0,85)', 'rgb(0,0,170)', 'rgb(0,0,255)', 'rgb(85,0,0)', 'rgb(85,0,85)', 'rgb(85,0,170)', 'rgb(85,0,255)', 'rgb(170,0,0)', 'rgb(170,0,85)', 'rgb(170,0,170)', 'rgb(170,0,255)', 'rgb(255,0,0)', 'rgb(255,0,85)', 'rgb(255,0,170)', 'rgb(255,0,255)', 'rgb(0,85,0)', 'rgb(0,85,85)', 'rgb(0,85,170)', 'rgb(0,85,255)', 'rgb(85,85,0)', 'rgb(85,85,85)', 'rgb(85,85,170)', 'rgb(85,85,255)', 'rgb(170,85,0)', 'rgb(170,85,85)', 'rgb(170,85,170)', 'rgb(170,85,255)', 'rgb(255,85,0)', 'rgb(255,85,85)', 'rgb(255,85,170)', 'rgb(255,85,255)', 'rgb(0,170,0)', 'rgb(0,170,85)', 'rgb(0,170,170)', 'rgb(0,170,255)', 'rgb(85,170,0)', 'rgb(85,170,85)', 'rgb(85,170,170)', 'rgb(85,170,255)', 'rgb(170,170,0)', 'rgb(170,170,85)', 'rgb(170,170,170)', 'rgb(170,170,255)', 'rgb(255,170,0)', 'rgb(255,170,85)', 'rgb(255,170,170)', 'rgb(255,170,255)', 'rgb(0,255,0)', 'rgb(0,255,85)', 'rgb(0,255,170)', 'rgb(0,255,255)', 'rgb(85,255,0)', 'rgb(85,255,85)', 'rgb(85,255,170)', 'rgb(85,255,255)', 'rgb(170,255,0)', 'rgb(170,255,85)', 'rgb(170,255,170)', 'rgb(170,255,255)', 'rgb(255,255,0)', 'rgb(255,255,85)', 'rgb(255,255,170)', 'rgb(255,255,255)'],
            shades: 0,
            hues: 4,
        });
    });

    $('#img_1').click(generate_similar(1));
    $('#img_2').click(generate_similar(2));
    $('#img_3').click(generate_similar(3));
    $('#img_4').click(generate_similar(4));
    $('#img_5').click(generate_similar(5));
    $('#img_6').click(generate_similar(6));
    $('#img_7').click(generate_similar(7));
    $('#img_8').click(generate_similar(8));
    $('#img_9').click(generate_similar(9));
    $('#btn_random').click(generate_random());



});
