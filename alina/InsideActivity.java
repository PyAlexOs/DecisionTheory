package com.example.kursovaya;

import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.os.Bundle;
import android.speech.AlternativeSpan;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import com.example.kursovaya.sqlite.DataBase;

public class InsideActivity extends AppCompatActivity {
    EditText art_input, name_input, about_input, count_input, price_input;

    Button btn_delete;
    String name, art, about, count, price, id;

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_inside);

        btn_delete = findViewById(R.id.btn_delete);
        name_input = findViewById(R.id.name_input);
        art_input = findViewById(R.id.art_input);
        about_input = findViewById(R.id.about_input);
        count_input = findViewById(R.id.count_input);
        price_input = findViewById(R.id.price_input);

        getset();

        btn_delete.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                DelDialog();
            }
        });
    }

    void DelDialog(){
        AlertDialog.Builder build = new AlertDialog.Builder(this);
        build.setMessage("Удалить "+name+" ?");
        build.setPositiveButton("Да", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int i) {
                DataBase db = new DataBase(InsideActivity.this);

                int count = db.readAll().getCount(); // количество записей в "виде" на бд, отображается вместо кнопки нет

                db.deleteRow(id);
                finish();
            }
        });
        build.setNegativeButton(count.toString(), new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int i) {

            }
        });
        build.create().show();
    }


    void getset(){
        if (getIntent().hasExtra("id") && getIntent().hasExtra("name") &&
                getIntent().hasExtra("art") && getIntent().hasExtra("count") &&
                getIntent().hasExtra("price") && getIntent().hasExtra("about")){

            //get
            id = getIntent().getStringExtra("id");
            name = getIntent().getStringExtra("name");
            art = getIntent().getStringExtra("art");
            count = getIntent().getStringExtra("count");
            price = getIntent().getStringExtra("price");
            about = getIntent().getStringExtra("about");

            //set
            art_input.setText(art);
            name_input.setText(name);
            about_input.setText(about);
            count_input.setText(count);
            price_input.setText(price);
            Log.d("InsideActivity", art+" "+name+" "+about+" "+count+" "+price);

        }else{
            Toast.makeText(this, "Нет данных", Toast.LENGTH_SHORT).show();
        }

    }

}