package com.example.kursovaya.sqlite;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.widget.Toast;
import androidx.annotation.Nullable;
import android.util.Log;

public class DataBase extends SQLiteOpenHelper{
    private Context context;
    private static final String DATABASE_NAME = "Assortiment.db";
    private static final int DATABASE_VERSION = 1;
    private static final String TABLE_NAME = "instrs";
    private static final String COLUMN_ID = "_id";
    private static final String COLUMN_ART = "art"; //артикул
    private static final String COLUMN_NAME = "name"; //название
    private static final String COLUMN_ABOUT = "about"; //описание
    private static final String COLUMN_COUNT = "count"; //количество на складе
    private static final String COLUMN_PRICE = "price"; //цена

    public DataBase(@Nullable Context context) {
        super(context, DATABASE_NAME, null, DATABASE_VERSION);
        this.context = context;
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        String q = "CREATE TABLE " + TABLE_NAME +
                " (" + COLUMN_ID + " INTEGER PRIMARY KEY AUTOINCREMENT, " +
                COLUMN_NAME + " TEXT, " +
                COLUMN_ART + " INTEGER, " +
                COLUMN_ABOUT + " TEXT, " +
                COLUMN_COUNT + " INTEGER, " +
                COLUMN_PRICE + " INTEGER);";
        db.execSQL(q);
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int i, int i1) {
        db.execSQL("DROP TABLE IF EXISTS " + TABLE_NAME);
        onCreate(db);
    }

    public void AddPos(int art, String name, String about, int count, int price){
        SQLiteDatabase db = this.getWritableDatabase();
        ContentValues cv = new ContentValues();

        cv.put(COLUMN_NAME, name);
        cv.put(COLUMN_ART, art);
        cv.put(COLUMN_PRICE, price);
        cv.put(COLUMN_COUNT, count);
        cv.put(COLUMN_ABOUT, about);

        long result = db.insert(TABLE_NAME, null, cv);
        if (result == -1){
            Toast.makeText(context, "Ошибка", Toast.LENGTH_SHORT).show();
        } else {
            Toast.makeText(context, "Успешно", Toast.LENGTH_SHORT).show();
        }
    }

    public Cursor readAll(){ //метод чтения данных в бд
        String q = "SELECT * FROM " + TABLE_NAME;
        SQLiteDatabase db = this.getReadableDatabase();
        Cursor cursor = null;
        if (db != null){
            cursor = db.rawQuery(q, null);
        }
        return cursor;
    }

    public void deleteRow(String id_row){
        SQLiteDatabase db = this.getWritableDatabase();
        long del_res = db.delete(TABLE_NAME, "_id=?", new String[]{id_row});
        if (del_res == -1){
            Toast.makeText(context,"Ошибка",Toast.LENGTH_SHORT).show();
        }else{
            Toast.makeText(context,"Успешно",Toast.LENGTH_SHORT).show();
        }
    }

}